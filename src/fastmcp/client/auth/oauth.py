from __future__ import annotations

import asyncio
import json
import webbrowser
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin, urlparse, parse_qs, urlencode

import anyio
import httpx
from mcp.client.auth import OAuthClientProvider as _MCPOAuthClientProvider
from mcp.client.auth import TokenStorage
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
)
from mcp.shared.auth import (
    OAuthMetadata as _MCPServerOAuthMetadata,
)
from mcp.shared.auth import (
    OAuthToken as OAuthToken,
)
from pydantic import AnyHttpUrl, ValidationError
import secrets
import hashlib
import base64

from fastmcp import settings as fastmcp_global_settings
from fastmcp.client.oauth_callback import (
    create_oauth_callback_server,
)
from fastmcp.utilities.http import find_available_port
from fastmcp.utilities.logging import get_logger

__all__ = ["OAuth"]

logger = get_logger(__name__)

def default_cache_dir() -> Path:
    return fastmcp_global_settings.home / "oauth-mcp-client-cache"

# Flexible OAuth models for real-world compatibility
class ServerOAuthMetadata(_MCPServerOAuthMetadata):
    """
    More flexible OAuth metadata model that accepts broader ranges of values
    than the restrictive MCP standard model.
    """
    code_challenge_methods_supported: list[str] | None = None
    token_endpoint_auth_methods_supported: list[str] | None = None
    grant_types_supported: list[str] | None = None
    response_types_supported: list[str] = ["code"]
    response_modes_supported: list[str] | None = None

class OAuthClientProvider(_MCPOAuthClientProvider):
    """
    OAuth client provider with flexible OAuth metadata discovery and PKCE support.
    """
    def __init__(
        self,
        server_url: str,
        client_metadata: OAuthClientMetadata,
        storage: TokenStorage,
        redirect_handler: callable,
        callback_handler: callable,
        code_verifier: str | None = None,
        skip_registration: bool = False,
    ):
        super().__init__(
            server_url=server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )
        self.code_verifier = code_verifier
        self.skip_registration = skip_registration

    async def _discover_oauth_metadata(
        self, server_url: str
    ) -> ServerOAuthMetadata | None:
        """
        Discover OAuth metadata with flexible validation and PKCE support check.
        """
        auth_base_url = self._get_authorization_base_url(server_url)
        url = urljoin(auth_base_url, "/.well-known/oauth-authorization-server")
        from mcp.types import LATEST_PROTOCOL_VERSION
        headers = {"MCP-Protocol-Version": LATEST_PROTOCOL_VERSION}
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                metadata_json = response.json()
                metadata = ServerOAuthMetadata.model_validate(metadata_json)
                # Check PKCE support if code_verifier is set
                if self.code_verifier and (
                    not metadata.code_challenge_methods_supported
                    or "S256" not in metadata.code_challenge_methods_supported
                ):
                    logger.warning("Server does not support PKCE (S256). Falling back to non-PKCE flow.")
                    self.code_verifier = None
                    self.client_metadata.token_endpoint_auth_method = "client_secret_post"
                    self.skip_registration = False
                logger.debug(f"OAuth metadata discovered: {metadata_json}")
                return metadata
            except Exception:
                try:
                    response = await client.get(url)
                    if response.status_code == 404:
                        return None
                    response.raise_for_status()
                    metadata_json = response.json()
                    metadata = ServerOAuthMetadata.model_validate(metadata_json)
                    if self.code_verifier and (
                        not metadata.code_challenge_methods_supported
                        or "S256" not in metadata.code_challenge_methods_supported
                    ):
                        logger.warning("Server does not support PKCE (S256). Falling back to non-PKCE flow.")
                        self.code_verifier = None
                        self.client_metadata.token_endpoint_auth_method = "client_secret_post"
                        self.skip_registration = False
                    logger.debug(
                        f"OAuth metadata discovered (no MCP header): {metadata_json}"
                    )
                    return metadata
                except Exception:
                    logger.exception("Failed to discover OAuth metadata")
                    return None

    async def _build_authorization_url(self, state: str, redirect_uri: str) -> str:
        """
        Build the authorization URL with PKCE support if code_verifier is set.
        """
        auth_url = await super()._build_authorization_url(state, redirect_uri)
        if self.code_verifier:
            code_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(self.code_verifier.encode()).digest()
            ).decode().rstrip("=")
            parsed_url = urlparse(auth_url)
            query = dict(parse_qs(parsed_url.query))
            query.update({"code_challenge": code_challenge, "code_challenge_method": "S256"})
            auth_url = parsed_url._replace(query=urlencode(query, doseq=True)).geturl()
        return auth_url

    async def _exchange_code_for_token(self, code: str, redirect_uri: str) -> OAuthToken:
        """
        Exchange authorization code for tokens, including code_verifier for PKCE.
        """
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        }
        if self.code_verifier:
            data["code_verifier"] = self.code_verifier
            if self.client_metadata.client_id:
                data["client_id"] = self.client_metadata.client_id
        else:
            client_info = await self.storage.get_client_info()
            if client_info:
                data["client_id"] = client_info.client_id
                data["client_secret"] = client_info.client_secret
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            return OAuthToken.model_validate(response.json())

    async def _register_client(self) -> OAuthClientInformationFull:
        """
        Register client with the server, unless skip_registration is True.
        """
        if self.skip_registration:
            logger.debug("Skipping client registration due to static client_id and PKCE")
            return OAuthClientInformationFull(
                client_id=self.client_metadata.client_id,
                client_secret=None,
                registration_access_token=None,
                registration_client_uri=None,
            )
        logger.debug("Performing dynamic client registration")
        return await super()._register_client()

class FileTokenStorage(TokenStorage):
    """
    File-based token storage implementation for OAuth credentials and tokens.
    Implements the mcp.client.auth.TokenStorage protocol.
    """
    def __init__(self, server_url: str, cache_dir: Path | None = None):
        self.server_url = server_url
        self.cache_dir = cache_dir or default_cache_dir()
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def get_base_url(url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def get_cache_key(self) -> str:
        base_url = self.get_base_url(self.server_url)
        return (
            base_url.replace("://", "_")
            .replace(".", "_")
            .replace("/", "_")
            .replace(":", "_")
        )

    def _get_file_path(self, file_type: Literal["client_info", "tokens"]) -> Path:
        key = self.get_cache_key()
        return self.cache_dir / f"{key}_{file_type}.json"

    async def get_tokens(self) -> OAuthToken | None:
        path = self._get_file_path("tokens")
        try:
            tokens = OAuthToken.model_validate_json(path.read_text())
            return tokens
        except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
            logger.debug(
                f"Could not load tokens for {self.get_base_url(self.server_url)}: {e}"
            )
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        path = self._get_file_path("tokens")
        path.write_text(tokens.model_dump_json(indent=2))
        logger.debug(f"Saved tokens for {self.get_base_url(self.server_url)}")

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        path = self._get_file_path("client_info")
        try:
            client_info = OAuthClientInformationFull.model_validate_json(
                path.read_text()
            )
            tokens = await self.get_tokens()
            if tokens is None:
                logger.debug(
                    f"No tokens found for client info at {self.get_base_url(self.server_url)}. "
                    "OAuth flow may have been incomplete. Clearing client info to force fresh registration."
                )
                client_info_path = self._get_file_path("client_info")
                client_info_path.unlink(missing_ok=True)
                return None
            return client_info
        except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
            logger.debug(
                f"Could not load client info for {self.get_base_url(self.server_url)}: {e}"
            )
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        path = self._get_file_path("client_info")
        path.write_text(client_info.model_dump_json(indent=2))
        logger.debug(f"Saved client info for {self.get_base_url(self.server_url)}")

    def clear(self) -> None:
        file_types: list[Literal["client_info", "tokens"]] = ["client_info", "tokens"]
        for file_type in file_types:
            path = self._get_file_path(file_type)
            path.unlink(missing_ok=True)
        logger.info(f"Cleared OAuth cache for {self.get_base_url(self.server_url)}")

    @classmethod
    def clear_all(cls, cache_dir: Path | None = None) -> None:
        cache_dir = cache_dir or default_cache_dir()
        if not cache_dir.exists():
            return
        file_types: list[Literal["client_info", "tokens"]] = ["client_info", "tokens"]
        for file_type in file_types:
            for file in cache_dir.glob(f"*_{file_type}.json"):
                file.unlink(missing_ok=True)
        logger.info("Cleared all OAuth client cache data.")

async def discover_oauth_metadata(
    server_base_url: str, httpx_kwargs: dict[str, Any] | None = None
) -> _MCPServerOAuthMetadata | None:
    well_known_url = urljoin(server_base_url, "/.well-known/oauth-authorization-server")
    logger.debug(f"Discovering OAuth metadata from: {well_known_url}")
    async with httpx.AsyncClient(**(httpx_kwargs or {})) as client:
        try:
            response = await client.get(well_known_url, timeout=10.0)
            if response.status_code == 200:
                logger.debug("Successfully discovered OAuth metadata")
                return _MCPServerOAuthMetadata.model_validate(response.json())
            elif response.status_code == 404:
                logger.debug(
                    "OAuth metadata not found (404) - server may not require auth"
                )
                return None
            else:
                logger.warning(f"OAuth metadata request failed: {response.status_code}")
                return None
        except (httpx.RequestError, json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"OAuth metadata discovery failed: {e}")
            return None

async def check_if_auth_required(
    mcp_url: str, httpx_kwargs: dict[str, Any] | None = None
) -> bool:
    async with httpx.AsyncClient(**(httpx_kwargs or {})) as client:
        try:
            response = await client.get(mcp_url, timeout=5.0)
            if response.status_code in (401, 403):
                return True
            if "WWW-Authenticate" in response.headers:
                return True
            return False
        except httpx.RequestError:
            return True

def generate_pkce_pair() -> tuple[str, str]:
    """
    Generate a PKCE code verifier and code challenge (S256 method).
    Returns: (code_verifier, code_challenge)
    """
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode().rstrip("=")
    return code_verifier, code_challenge

def OAuth(
    mcp_url: str,
    scopes: str | list[str] | None = None,
    client_name: str = "FastMCP Client",
    token_storage_cache_dir: Path | None = None,
    additional_client_metadata: dict[str, Any] | None = None,
    use_pkce: bool = False,
    static_client_id: str | None = None,
) -> _MCPOAuthClientProvider:
    """
    Create an OAuthClientProvider for an MCP server with optional PKCE support and static client_id.

    Args:
        mcp_url: Full URL to the MCP endpoint (e.g. "http://host/mcp/sse/")
        scopes: OAuth scopes to request. Can be a space-separated string or a list of strings.
        client_name: Name for this client during registration
        token_storage_cache_dir: Directory for FileTokenStorage
        additional_client_metadata: Extra fields for OAuthClientMetadata
        use_pkce: If True, use PKCE for public client authentication
        static_client_id: Static client_id to use without registration (requires use_pkce=True)

    Returns:
        OAuthClientProvider
    """
    parsed_url = urlparse(mcp_url)
    server_base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

    # Validate static_client_id usage
    if static_client_id and not use_pkce:
        raise ValueError("static_client_id can only be used with use_pkce=True")

    # Setup OAuth client
    redirect_port = find_available_port()
    redirect_uri = f"http://127.0.0.1:{redirect_port}/callback"

    if isinstance(scopes, list):
        scopes = " ".join(scopes)

    # Generate PKCE parameters if enabled
    code_verifier = None
    if use_pkce:
        code_verifier, _ = generate_pkce_pair()

    # Set token_endpoint_auth_method to "none" for PKCE
    token_endpoint_auth_method = "none" if use_pkce else "client_secret_post"

    # Use static client_id if provided and PKCE is enabled
    client_metadata = OAuthClientMetadata(
        client_name=client_name,
        client_id=static_client_id if use_pkce and static_client_id else None,
        redirect_uris=[AnyHttpUrl(redirect_uri)],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method=token_endpoint_auth_method,
        scope=scopes,
        **(additional_client_metadata or {}),
    )

    # Create server-specific token storage
    storage = FileTokenStorage(
        server_url=server_base_url, cache_dir=token_storage_cache_dir
    )

    # Define OAuth handlers
    async def redirect_handler(authorization_url: str) -> None:
        """Open browser for authorization."""
        logger.info(f"OAuth authorization URL: {authorization_url}")
        webbrowser.open(authorization_url)

    async def callback_handler() -> tuple[str, str | None]:
        """Handle OAuth callback and return (auth_code, state)."""
        response_future = asyncio.get_running_loop().create_future()
        server = create_oauth_callback_server(
            port=redirect_port,
            server_url=server_base_url,
            response_future=response_future,
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(server.serve)
            logger.info(
                f"🎧 OAuth callback server started on http://127.0.0.1:{redirect_port}"
            )
            TIMEOUT = 300.0
            try:
                with anyio.fail_after(TIMEOUT):
                    auth_code, state = await response_future
                    return auth_code, state
            except TimeoutError:
                raise TimeoutError(f"OAuth callback timed out after {TIMEOUT} seconds")
            finally:
                server.should_exit = True
                await asyncio.sleep(0.1)
                tg.cancel_scope.cancel()

    # Create OAuth provider
    oauth_provider = OAuthClientProvider(
        server_url=server_base_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        code_verifier=code_verifier,
        skip_registration=use_pkce and static_client_id is not None,
    )

    return oauth_provider
