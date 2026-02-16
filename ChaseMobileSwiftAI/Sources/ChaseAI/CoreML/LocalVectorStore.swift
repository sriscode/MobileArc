
// LocalVectorStore.swift
// On-device semantic search over transactions using Core ML MiniLM embeddings
// Enables Foundation Models to find relevant past transactions without cloud round-trips
// Train model: MLTraining/embeddings/convert_minilm.py

import CoreML
import Foundation

// MARK: - Vector Store (Actor for thread safety)
//LocalVectorStore for persistant vector db.
actor LocalVectorStore {
    private var entries: [VectorEntry] = []
    private var embedder: MiniLMEmbeddingRunner?
    private(set) var isReady = false

    // MARK: - Init

    func initialize() throws {
        embedder = try MiniLMEmbeddingRunner()
        isReady  = true
        print("✅ LocalVectorStore ready")
    }

    // MARK: - Index Transactions
    // Called on every background account sync to keep the store fresh

    func indexTransactions(_ transactions: [Transaction]) throws {
        guard let embedder = embedder else { return }
        for txn in transactions {
            let text      = "\(txn.merchant) \(txn.category) \(txn.formattedAmount) \(txn.date)"
            let embedding = try embedder.embed(text)
            let entry     = VectorEntry(
                id:        "txn:\(txn.id)",
                vector:    embedding,
                metadata:  [
                    "merchant": txn.merchant,
                    "amount":   txn.formattedAmount,
                    "category": txn.category,
                    "date":     txn.date
                ]
            )
            entries.removeAll { $0.id == entry.id }
            entries.append(entry)
        }
    }

    // MARK: - Semantic Search

    func search(query: String, topK: Int = 8) throws -> [VectorMatch] {
        guard let embedder = embedder else { return [] }
        let queryVec = try embedder.embed(query)
        return entries
            .map { VectorMatch(entry: $0, score: embedder.cosineDistance(queryVec, $0.vector)) }
            .sorted { $0.score < $1.score }
            .prefix(topK)
            .map { $0 }
    }

    // MARK: - RAG Context Builder
    // Call this to enrich Foundation Models prompts with relevant transactions

    func buildRAGContext(for query: String) throws -> String {
        let matches = try search(query: query, topK: 5)
        guard !matches.isEmpty else { return "" }
        let lines = matches.map {
            "\($0.entry.metadata["date"] ?? "") | \($0.entry.metadata["merchant"] ?? "") | \($0.entry.metadata["amount"] ?? "") | \($0.entry.metadata["category"] ?? "")"
        }.joined(separator: "\n")
        return "Relevant past transactions:\n\(lines)"
    }
}

// MARK: - Vector Entry & Match

struct VectorEntry {
    let id:       String
    let vector:   [Float]
    let metadata: [String: String]
}

struct VectorMatch {
    let entry: VectorEntry
    let score: Float    // lower = more similar (cosine distance)
}

// MARK: - MiniLM Embedding Runner
// Named MiniLMEmbeddingRunner (not MiniLMEmbedder) to avoid clash with the
// Swift class Xcode auto-generates from MiniLMEmbedder.mlpackage.
// We use the auto-generated MiniLMEmbedder class directly for inference.

class MiniLMEmbeddingRunner {
    // Xcode auto-generates `MiniLMEmbedder` from the .mlpackage —
    // we hold a reference to that generated class here.
    private let model:     MiniLMEmbedder
    private let tokenizer: LightweightBertTokenizer
    private let dim       = 384
    private let maxLen    = 64

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        // MiniLMEmbedder() is the Xcode-generated initialiser from the .mlpackage
        model     = try MiniLMEmbedder(configuration: config)
        tokenizer = LightweightBertTokenizer()
    }

    func embed(_ text: String) throws -> [Float] {
        let tokens = tokenizer.encode(text, maxLength: maxLen)

        let inputIds = try MLMultiArray(shape: [1, maxLen as NSNumber], dataType: .int32)
        let attnMask = try MLMultiArray(shape: [1, maxLen as NSNumber], dataType: .int32)
        for i in 0..<tokens.inputIds.count {
            inputIds[i] = NSNumber(value: tokens.inputIds[i])
            attnMask[i] = NSNumber(value: tokens.attentionMask[i])
        }

        // Use the auto-generated typed prediction API
        let input  = MiniLMEmbedderInput(input_ids: inputIds, attention_mask: attnMask)
        let output = try model.prediction(input: input)

        let emb = output.embeddings   // MLMultiArray — auto-generated output property
        return (0..<dim).map { Float(truncating: emb[$0]) }
    }

    func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        let dot   = zip(a, b).map(*).reduce(0, +)
        let normA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let normB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        guard normA > 0, normB > 0 else { return 1.0 }
        return 1.0 - (dot / (normA * normB))
    }
}

enum EmbedderError: Error {
    case inferenceFailure
}
