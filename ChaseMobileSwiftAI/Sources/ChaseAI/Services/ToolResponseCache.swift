//
//  ToolResponseCache.swift.swift
//  ChaseMobileSwiftAI
//
//  Created by shaurya on 2/15/26.
//

// ToolResponseCache.swift
import Foundation

actor ToolResponseCache {

    static let shared = ToolResponseCache()

    private struct CacheEntry {
        let value:     String
        let fetchedAt: Date
        let ttl:       TimeInterval
        var isExpired: Bool { Date().timeIntervalSince(fetchedAt) > ttl }
    }

    private var store: [String: CacheEntry] = [:]

    // TTL per tool — how fresh does this data need to be?
    enum TTL {
        static let accountBalance:   TimeInterval = 30          // 30 seconds
        static let transactions:     TimeInterval = 60          // 1 minute
        static let savingsRates:     TimeInterval = 3_600       // 1 hour — rates rarely change
        static let creditScore:      TimeInterval = 86_400      // 24 hours — updates monthly
    }

    func get(_ key: String) -> String? {
        guard let entry = store[key], !entry.isExpired else {
            store.removeValue(forKey: key)   // evict expired entry
            return nil
        }
        return entry.value
    }

    func set(_ key: String, value: String, ttl: TimeInterval) {
        store[key] = CacheEntry(value: value, fetchedAt: Date(), ttl: ttl)
    }

    // Call this after a transfer executes — balance is now stale
    func invalidate(_ key: String) {
        store.removeValue(forKey: key)
    }

    // Call on session reset or logout
    func invalidateAll() {
        store.removeAll()
    }

    // Invalidate all entries for a specific tool
    func invalidateTool(_ toolName: String) {
        store = store.filter { !$0.key.hasPrefix(toolName) }
    }
}
