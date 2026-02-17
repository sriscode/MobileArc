//
//  Untitled.swift
//  ChaseMobileSwiftAI
//
//  Created by shaurya on 2/16/26.
//


/*
ident classifier
 
 
 
 DistilBERT vs TF-IDF + Logistic RegressionTF-IDF + Logistic RegressionDistilBERTCategoryClassical MLDeep Learning / Neural Network AIArchitectureLinear model (1 layer)Transformer (66 million parameters, 6 layers)Training approachTrained on your 157 examples onlyPre-trained on billions of words, then fine-tunedWhat it learnsWord weights for your 8 classesLanguage understanding + your 8 classesModel typeDiscriminative (direct classification)Representation learning (understands semantics)Size300 KB133 MB (or 8 MB quantized)Feels like "AI"?❌ Feels like statistics✅ Feels like language understanding
 
 
 */


/*
 prompt
 
 If user says hello/hi/thanks after a query:
            - Respond briefly: "Hello! How can I help you?"
            - Do NOT continue previous topic
            - Do NOT mention transfers or actions
            - Do NOT repeat previous information
            
            TOOL USAGE:
            
            stageTransfer - ONLY call when:
            - User explicitly requests a transfer with amount
            - User says "transfer $200 to savings" or similar
            - NEVER call for "hello", "thanks", "ok", or questions
 
 
 
 
 
 */

/*
 Fix 3 — Proactive Session Summarisation (Main Fix)
 When the session is getting long, summarise it before it overflows, then start fresh with the summary:
 
 
 // FoundationModelsService.swift

 // Summarise and reset before hitting the window limit
 private func handleContextOverflow(
     query:   String,
     intent:  UserIntent,
     context: AccountContext
 ) async throws -> AgentResponse {

     // Step 1 — generate a summary from the current session
     //           before it becomes unusable
     let summary = await summariseCurrentSession(context: context)

     // Step 2 — start a completely fresh session
     chatSession      = makeSummarisedSession(summary: summary)
     chatSessionTurns = 0
     contextInjected  = true   // summary already includes context

     guard let freshSession = chatSession else {
         throw AgentError.foundationModelsUnavailable
     }

     // Step 3 — retry the original query in the fresh session
     let security  = AppSecurityService()
     let response  = try await freshSession.respond(to: query)
     let safe      = await security.sanitizeAndAudit(response.content, userId: context.userId)
     return .text(safe)
 }

 // Use the analysis session (no tools, structured output)
 // to produce a compact summary of what was discussed
 private func summariseCurrentSession(context: AccountContext) async -> String {
     guard let session = chatSession else {
         return "No prior conversation context."
     }

     // Ask the session to summarise itself before we discard it
     let summaryPrompt = """
         In 3-5 bullet points, summarise the key facts established in this
         conversation so far. Include:
         - What the user asked about
         - Any specific amounts, dates, merchants, or accounts mentioned
         - Any conclusions or answers already given
         Be concise. This summary will be used to continue the conversation
         in a new session. Do not include any account numbers or card details.
         """

     do {
         let summary = try await session.respond(to: summaryPrompt)
         return summary.content
     } catch {
         // If even the summary fails (very full session), use minimal fallback
         return "Previous conversation covered financial queries about this account."
     }
 }

 // Start a fresh session that knows about the prior conversation
 private func makeSummarisedSession(summary: String) -> LanguageModelSession {
     LanguageModelSession(
         instructions: Instructions("""
             You are Chase AI, a financial assistant inside the Chase mobile app.
             You only discuss topics related to the user's Chase accounts.

             CONVERSATION CONTEXT FROM PRIOR SESSION:
             \(summary)

             The user is continuing their conversation. Use the context above
             to answer follow-up questions correctly.

             TOOL USAGE RULES:
             - Call getAccountBalance ONLY when asked about balances
             - Call getTransactions ONLY when asked about transaction history
             - Call stageTransfer ONLY for explicit new transfer requests
             - Call getSavingsRates ONLY when asked about rates
             - Call getCreditScore ONLY when asked about credit health
             - Do NOT call tools for date, time, or general questions
             """),
         tools: bankingTools
     )
 }
 
 
 
 
 
 
 
 
 
 
 
 
 
 */
