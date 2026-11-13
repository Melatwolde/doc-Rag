# Authentication and Session Management Service

## JWT Architecture and Token Lifecycle
The authentication subsystem issues short-lived JSON Web Tokens (JWT) for stateless API access and long-lived refresh tokens stored securely in HTTP-only cookies. Access tokens have an expiration lifetime configured via the CLI flag `--token-expiry-seconds` (defaults to 3600 seconds / 60 minutes).

When an access token passes its expiration window, API gateways reject the request and immediately raise the error code `ERR_AUTH_TOKEN_EXPIRED`. The client must then exchange its refresh token before retrying.

```json
{
  "error": "ERR_AUTH_TOKEN_EXPIRED",
  "message": "Access token expired. Please refresh session credentials.",
  "status_code": 401
}
```

## Refresh Token Exchange Endpoint
To renew session credentials without prompting the user to enter passwords again, client applications dispatch an authenticated request to `POST /api/v1/auth/refresh`.

The core function `refresh_access_token` verifies the cryptographically signed refresh token against the rotation whitelist:

```python
def refresh_access_token(refresh_token: str) -> dict:
    """Validate refresh token and issue a newly signed access token."""
    session = validate_session_whitelist(refresh_token)
    if not session.is_valid:
        raise SecurityException("ERR_INVALID_SIGNATURE", "Invalid or revoked refresh token")
    new_token = generate_jwt(user_id=session.user_id, ttl=3600)
    return {"access_token": new_token, "expires_in": 3600}
```

If cryptographic signature verification fails or tampering is detected, the gateway emits `ERR_INVALID_SIGNATURE` and invalidates the entire session hierarchy.

## Security Configuration and Secrets
Token signing and asymmetric verification use the key defined by `auth.jwt.secret_key` in `config.yaml`. Supported signing algorithms are defined under `auth.jwt.algorithm`, with `HS256` as the default symmetric algorithm and `RS256` for asymmetric multi-service architectures. Never commit `auth.jwt.secret_key` to version control.
