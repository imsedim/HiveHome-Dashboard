import base64
import json
from pathlib import Path
from typing import Literal
import pycognito
from pycognito.exceptions import MFAChallengeException
from pycognito.aws_srp import hash_sha256, pad_hex, get_random, hex_to_long, hex_hash, G_HEX, N_HEX, AWSSRP as srp
import boto3
import os
import platform

_ = MFAChallengeException


class AWSSRP(srp):
    challenge_username: str = None

    def process_challenge(self, challenge_parameters, request_parameters):
        self.challenge_username = challenge_parameters.get("USERNAME")
        return super().process_challenge(challenge_parameters, request_parameters)

    # def authenticate_device(self):
    #     boto_client = self.client
    #     auth_params = self.get_auth_params()
    #     auth_params["DEVICE_KEY"] = device_key

    #     response = boto_client.initiate_auth(
    #         AuthFlow="USER_SRP_AUTH",
    #         AuthParameters=auth_params,
    #         ClientId=self.client_id,
    #     )
    #     if response["ChallengeName"] == self.PASSWORD_VERIFIER_CHALLENGE:
    #         challenge_response = self.process_challenge(
    #             response["ChallengeParameters"], auth_params
    #         )
    #         tokens = boto_client.respond_to_auth_challenge(
    #             ClientId=self.client_id,
    #             ChallengeName=self.PASSWORD_VERIFIER_CHALLENGE,
    #             ChallengeResponses=challenge_response,
    #             **dict(ClientMetadata=client_metadata) if client_metadata else {},
    #         )

    #         if tokens.get("ChallengeName") == self.NEW_PASSWORD_REQUIRED_CHALLENGE:
    #             raise ForceChangePasswordException(
    #                 "Change password before authenticating"
    #             )

    #         if tokens.get("ChallengeName") == self.SMS_MFA_CHALLENGE:
    #             raise SMSMFAChallengeException("Do SMS MFA", tokens)

    #         if tokens.get("ChallengeName") == self.SOFTWARE_TOKEN_MFA_CHALLENGE:
    #             raise SoftwareTokenMFAChallengeException(
    #                 "Do Software Token MFA", tokens
    #             )

    #         return tokens

    #     raise NotImplementedError(
    #         f"The {response['ChallengeName']} challenge is not supported"
    #     )


class Cognito(pycognito.Cognito):
    challenge_username: str = None

    def __init__(self, user_pool_id, client_id, user_pool_region=None, username=None, id_token=None, refresh_token=None, access_token=None, client_secret=None, access_key=None, secret_key=None, session=None, botocore_config=None, boto3_client_kwargs=None, device_key=None, device_group_key=None):
        super().__init__(user_pool_id, client_id, user_pool_region, username, id_token, refresh_token, access_token, client_secret, access_key, secret_key, session, botocore_config, boto3_client_kwargs)
        self.device_key = device_key
        self.device_group_key = device_group_key

    def _set_tokens(self, tokens):
        if "NewDeviceMetadata" in tokens["AuthenticationResult"]:
            self.device_key = tokens["AuthenticationResult"]["NewDeviceMetadata"]["DeviceKey"]
            self.device_group_key = tokens["AuthenticationResult"]["NewDeviceMetadata"]["DeviceGroupKey"]
        return super()._set_tokens(tokens)

    def confirm_device(self):
        device_password, device_secret_verifier_config = generate_hash_device(self.device_group_key, self.device_key)

        params = dict(AccessToken=self.access_token,
                      DeviceKey=self.device_key,
                      DeviceSecretVerifierConfig=device_secret_verifier_config,
                      DeviceName=platform.node())

        response = boto3.client("cognito-idp", region_name=self.user_pool_region).confirm_device(**params)

        return response, device_password

    def update_device_status(self, remembered: bool):
        client = boto3.client("cognito-idp", region_name=self.user_pool_region)
        client.update_device_status(AccessToken=self.access_token,
                                    DeviceKey=self.device_key,
                                    DeviceRememberedStatus=["not_remembered", "remembered"][remembered])

    def authenticate(self, password, client_metadata=None):
        """
        Authenticate the user using the SRP protocol
        :param password: The user's passsword
        :param client_metadata: Metadata you can provide for custom workflows that RespondToAuthChallenge triggers.
        :return:
        """
        aws = AWSSRP(username=self.username,
                     password=password,
                     pool_id=self.user_pool_id,
                     client_id=self.client_id,
                     client=self.client,
                     client_secret=self.client_secret)
        try:
            tokens = aws.authenticate_user(client_metadata=client_metadata)
        except MFAChallengeException as mfa_challenge:
            self.mfa_tokens = mfa_challenge.get_tokens()
            self.mfa_username = aws.challenge_username
            raise mfa_challenge
        else:
            self._set_tokens(tokens)

    # def refresh_authentication(tokens: AuthenticationResult) -> AuthenticationResult:
    #     username = jwt.decode(tokens.id_token, verify=False)
    #     import boto3
    #     client = boto3.client("cognito-idp", region_name="eu-west-1")
    #     client_id = "3rl4i0ajrmtdm8sbre54p9dvd9"
    #     response = client.initiate_auth(
    #                 ClientId=client_id,
    #                 AuthFlow='REFRESH_TOKEN_AUTH',
    #                 AuthParameters={
    #                     'REFRESH_TOKEN': tokens.refresh_token,
    #                     'SECRET_HASH': AWSSRP.get_secret_hash(client_id, clientSecret, username["cognito:username"]),
    #                 })

    def renew_access_token(self):
        """
        Sets a new access token on the User using the cached refresh token.
        """
        auth_params = {"REFRESH_TOKEN": self.refresh_token,
                       "DEVICE_KEY": self.device_key}
        self._add_secret_hash(auth_params, "SECRET_HASH")

        refresh_response = self.client.initiate_auth(ClientId=self.client_id,
                                                     AuthFlow="REFRESH_TOKEN_AUTH",
                                                     AuthParameters=auth_params)

        self._set_tokens(refresh_response)

    def authenticate_device(self, password):
        """
        Authenticate the user using the SRP protocol
        :param password: The user's passsword
        :param client_metadata: Metadata you can provide for custom workflows that RespondToAuthChallenge triggers.
        :return:
        """
        aws = AWSSRP(username=self.username,
                     password=password,
                     pool_id=self.user_pool_id,
                     client_id=self.client_id,
                     client=self.client,
                     client_secret=self.client_secret)
        try:
            tokens = aws.authenticate_user(client_metadata=client_metadata)
        except MFAChallengeException as mfa_challenge:
            self.mfa_tokens = mfa_challenge.get_tokens()
            self.mfa_username = aws.challenge_username
            raise mfa_challenge
        else:
            self._set_tokens(tokens)


def generate_hash_device(device_group_key, device_key):
    # source: https://github.com/amazon-archives/amazon-cognito-identity-js/blob/6b87f1a30a998072b4d98facb49dcaf8780d15b0/src/AuthenticationHelper.js#L137

    # random device password, which will be used for DEVICE_SRP_AUTH flow
    device_password = base64.standard_b64encode(os.urandom(40)).decode('utf-8')

    combined_string = '%s%s:%s' % (device_group_key, device_key, device_password)
    combined_string_hash = hash_sha256(combined_string.encode('utf-8'))
    salt = pad_hex(get_random(16))

    x_value = hex_to_long(hex_hash(salt + combined_string_hash))
    g = hex_to_long(G_HEX)
    big_n = hex_to_long(N_HEX)
    verifier_device_not_padded = pow(g, x_value, big_n)
    verifier = pad_hex(verifier_device_not_padded)

    device_secret_verifier_config = {
        "PasswordVerifier": base64.standard_b64encode(bytearray.fromhex(verifier)).decode('utf-8'),
        "Salt": base64.standard_b64encode(bytearray.fromhex(salt)).decode('utf-8')
    }
    return device_password, device_secret_verifier_config
