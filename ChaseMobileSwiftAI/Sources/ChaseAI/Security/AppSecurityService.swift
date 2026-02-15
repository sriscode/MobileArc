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
        let rules: [(pattern: String, replacement: String)] = [

            // ── Identity ──────────────────────────────────────────────
            // SSN — formatted
            (#"\b\d{3}-\d{2}-\d{4}\b"#,                              "[SSN-REDACTED]"),
            // SSN — raw 9 digits (with word boundary context)
            (#"(?i)(ssn|social security)[:\s#]*\d{9}"#,              "[SSN-REDACTED]"),
            // Date of birth
            (#"(?i)(dob|date of birth|born on)[:\s]*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}"#, "[DOB-REDACTED]"),
            // Passport number
            (#"\b[A-Z]{1,2}\d{6,9}\b"#,                              "[PASSPORT-REDACTED]"),
            // Driver's license (US general format)
            (#"(?i)(driver.?s?\s+licen[sc]e|dl\s*#?)[:\s]*[A-Z0-9]{6,15}"#, "[DL-REDACTED]"),

            // ── Payment Credentials ───────────────────────────────────
            // Card numbers — 16 digit with optional separators
            (#"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"#,    "[CARD-REDACTED]"),
            // Amex — 15 digit
            (#"\b3[47]\d{2}[\s\-]?\d{6}[\s\-]?\d{5}\b"#,            "[CARD-REDACTED]"),
            // CVV — 3 or 4 digits after keyword
            (#"(?i)(cvv|cvc|cvv2|security code)[:\s]*\d{3,4}"#,      "[CVV-REDACTED]"),
            // PIN
            (#"(?i)(pin|passcode)[:\s]*\d{4,6}"#,                    "[PIN-REDACTED]"),
            // Expiry date on card context
            (#"(?i)(exp|expiry|expiration)[:\s]*\d{2}[\/\-]\d{2,4}"#,"[EXPIRY-REDACTED]"),

            // ── Account & Routing ─────────────────────────────────────
            // US routing numbers — exactly 9 digits (ABA format)
            (#"(?i)(routing|aba|rtn)[:\s#]*\d{9}"#,                  "[ROUTING-REDACTED]"),
            // Account numbers after keyword (8-17 digits)
            (#"(?i)(account\s*#?|acct)[:\s]*\d{8,17}"#,              "[ACCOUNT-REDACTED]"),
            // IBAN
            (#"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7,19}\b"#,              "[IBAN-REDACTED]"),
            // SWIFT/BIC code
            (#"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b"#,     "[SWIFT-REDACTED]"),

            // ── Authentication ────────────────────────────────────────
            // OTP / 2FA codes
            (#"(?i)(otp|one.?time|verification|auth)\s*(code|token)[:\s]*\d{4,8}"#, "[OTP-REDACTED]"),
            // Passwords after keyword
            (#"(?i)(password|passwd|pwd)[:\s]+\S+"#,                  "[PASSWORD-REDACTED]"),
            // Security questions/answers
            (#"(?i)(mother.?s maiden|first pet|childhood)[:\s]+\w+"#, "[SECURITY-ANS-REDACTED]"),

            // ── Wire Transfer Details ─────────────────────────────────
            // Wire amounts with recipient — high risk combination
            (#"(?i)wire\s+\$?\d[\d,]*\.?\d*\s+to"#,                  "[WIRE-REDACTED]"),

            // ── Contact (partial — banks vary on policy) ─────────────
            // Full phone numbers
            (#"\b(\+1[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b"#, "[PHONE-REDACTED]"),
        ]

        for rule in rules {
            s = s.replacingOccurrences(
                of: rule.pattern,
                with: rule.replacement,
                options: .regularExpression
            )
        }
        return s
    }
    
    func sanitizeAndAudit(_ text: String, userId: String) async -> String {
        let sanitized = sanitizeForLLM(text)

        // Only log if something was actually redacted
        if sanitized != text {
            let patterns = detectRedactedPatterns(original: text, sanitized: sanitized)

            // Fire-and-forget — never blocks the query
            Task.detached(priority: .background) {
                await AuditLogger.shared.log(
                    event: "pii_redacted_from_input",
                    metadata: [
                        "user_id":      userId,
                        "patterns":     patterns.joined(separator: ","),
                        "query_hash":   String(text.hashValue),  // hash only, never raw text
                        "pattern_count": String(patterns.count)
                    ]
                )
            }
        }
        return sanitized
    }

    // Helper — identifies which pattern types fired
    private func detectRedactedPatterns(original: String, sanitized: String) -> [String] {
        var found: [String] = []

        let checks: [(String, String)] = [
            ("[CARD-REDACTED]",     "card_number"),
            ("[SSN-REDACTED]",      "ssn"),
            ("[CVV-REDACTED]",      "cvv"),
            ("[PIN-REDACTED]",      "pin"),
            ("[ROUTING-REDACTED]",  "routing_number"),
            ("[ACCOUNT-REDACTED]",  "account_number"),
            ("[OTP-REDACTED]",      "otp"),
            ("[PASSWORD-REDACTED]", "password"),
            ("[PHONE-REDACTED]",    "phone_number"),
            ("[IBAN-REDACTED]",     "iban"),
            ("[DOB-REDACTED]",      "date_of_birth"),
        ]

        for (marker, name) in checks {
            if sanitized.contains(marker) { found.append(name) }
        }
        return found
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

