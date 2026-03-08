from __future__ import annotations

import asyncio
import base64
import os
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from pathlib import Path
from typing import Any

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from config import settings


class GmailTool:
    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    OPTIONAL_OPENID_SCOPES = [
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ]

    @classmethod
    def _all_scopes(cls) -> list[str]:
        return cls.SCOPES + cls.OPTIONAL_OPENID_SCOPES

    def _client_config(self) -> dict[str, Any]:
        return {
            "web": {
                "client_id": settings.gmail_client_id,
                "client_secret": settings.gmail_client_secret,
                "redirect_uris": [settings.gmail_redirect_uri],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }

    def _flow(self) -> Flow:
        return Flow.from_client_config(
            client_config=self._client_config(),
            scopes=self._all_scopes(),
            redirect_uri=settings.gmail_redirect_uri,
        )

    def get_auth_url(self) -> str:
        flow = self._flow()
        auth_url, _state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        return auth_url

    def exchange_code(self, code: str) -> dict[str, Any]:
        # Google can return extra OpenID scopes; relax strict scope validation.
        os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
        flow = self._flow()
        flow.fetch_token(code=code)
        creds = flow.credentials
        return {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes,
            "expiry": creds.expiry.isoformat() if creds.expiry else None,
        }

    @staticmethod
    def _build_credentials(credentials_dict: dict[str, Any]) -> Credentials:
        creds = Credentials.from_authorized_user_info(credentials_dict)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return creds

    @staticmethod
    def _build_message(to: str, subject: str, body: str, attachment_path: str | None = None) -> dict[str, str]:
        message = MIMEMultipart()
        message["to"] = to
        message["subject"] = subject
        message.attach(MIMEText(body, "plain"))

        if attachment_path and Path(attachment_path).exists():
            with open(attachment_path, "rb") as attachment_file:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment_file.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename="{Path(attachment_path).name}"',
                )
                message.attach(part)

        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return {"raw": encoded}

    @staticmethod
    def _send_sync(credentials_dict: dict[str, Any], to: str, subject: str, body: str, attachment_path: str | None) -> bool:
        creds = GmailTool._build_credentials(credentials_dict)
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        payload = GmailTool._build_message(to, subject, body, attachment_path)
        service.users().messages().send(userId="me", body=payload).execute()
        return True

    async def send_email(
        self,
        credentials_dict: dict[str, Any],
        to: str,
        subject: str,
        body: str,
        attachment_path: str | None = None,
    ) -> bool:
        try:
            return await asyncio.to_thread(
                self._send_sync,
                credentials_dict,
                to,
                subject,
                body,
                attachment_path,
            )
        except Exception as exc:
            print(f"Gmail send error: {exc}")
            return False
