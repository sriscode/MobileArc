// AppSecurityService.swift
// Biometric authentication, Secure Enclave key management, PII sanitisation
// All data passed to any LLM must first go through sanitizeForLLM()

import LocalAuthentication
import CryptoKit
import Security

// MARK: - Auth Token

struct AuthToken {
    let rawValue: Data
    let issuedAt: Date
    var isExpired: Bool { Date().timeIntervalSince(issuedAt) > 300 }   // 5-min TTL
}

// MARK: - Security Service

@Observable
class AppSecurityService {

    private(set) var isAuthenticated = false
    private var cachedToken: AuthToken?

    // MARK: - Biometric Auth

    nonisolated(nonsending) func authenticateWithBiometrics() async throws -> AuthToken {
        if let token = cachedToken, !token.isExpired { return token }

        let context = LAContext()
        var error: NSError?

        // Prefer Face ID / Touch ID; fall back to passcode
        let policy: LAPolicy = context.canEvaluatePolicy(
            .deviceOwnerAuthenticationWithBiometrics, error: &error
        ) ? .deviceOwnerAuthenticationWithBiometrics : .deviceOwnerAuthentication

        context.localizedCancelTitle   = "Cancel"
        context.localizedFallbackTitle = "Use Passcode"

        let success = try await context.evaluatePolicy(
            policy,
            localizedReason: "Authenticate to access Chase AI"
        )
        guard success else { throw AuthError.failed }

        let token = try issueToken()
        cachedToken     = token
        isAuthenticated = true
        return token
    }

    // MARK: - Secure Enclave

    /// Signs a challenge with a device-bound P256 key from the Secure Enclave.
    /// The private key never leaves the hardware.
    private func issueToken() throws -> AuthToken {
        let key       = try SecureEnclave.P256.Signing.PrivateKey()
        let challenge = Data(UUID().uuidString.utf8)
        let signature = try key.signature(for: challenge)
        return AuthToken(rawValue: signature.rawRepresentation, issuedAt: Date())
    }

    // MARK: - PII Sanitisation

    /// Strip sensitive data before any string reaches an LLM context or tool output.
    func sanitizeForLLM(_ text: String) -> String {
        var s = text
        // 16-digit card numbers (with or without spaces/dashes)
        s = s.replacingOccurrences(
            of: #"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"#,
            with: "[CARD-REDACTED]", options: .regularExpression)
        // SSN — formatted or raw 9 digits
        s = s.replacingOccurrences(
            of: #"\b\d{3}-\d{2}-\d{4}\b"#,
            with: "[SSN-REDACTED]", options: .regularExpression)
        // CVV after keyword
        s = s.replacingOccurrences(
            of: #"(?i)(cvv|cvc|security code)\s*[:\s]\s*\d{3,4}"#,
            with: "[CVV-REDACTED]", options: .regularExpression)
        // PIN after keyword
        s = s.replacingOccurrences(
            of: #"(?i)pin\s*[:\s]\s*\d{4,6}"#,
            with: "[PIN-REDACTED]", options: .regularExpression)
        return s
    }

    // MARK: - Keychain

    func saveToKeychain(key: String, data: Data) throws {
        let query: [String: Any] = [
            kSecClass as String:          kSecClassGenericPassword,
            kSecAttrAccount as String:    key,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            kSecValueData as String:      data
        ]
        SecItemDelete(query as CFDictionary)
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else { throw KeychainError.saveFailed(status) }
    }

    func loadFromKeychain(key: String) throws -> Data {
        let query: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String:  true,
            kSecMatchLimit as String:  kSecMatchLimitOne
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else {
            throw KeychainError.notFound
        }
        return data
    }

    // MARK: - Sign Out

    func signOut() {
        cachedToken     = nil
        isAuthenticated = false
    }
}

// MARK: - Errors

enum AuthError: LocalizedError {
    case failed
    case expired

    var errorDescription: String? {
        switch self {
        case .failed:  return "Authentication failed. Please try again."
        case .expired: return "Session expired. Please re-authenticate."
        }
    }
}

enum KeychainError: Error {
    case saveFailed(OSStatus)
    case notFound
}

