import os, json, re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

training = [
    # ── balance_query (25 examples) ──────────────────────────────────────────
    ("What is my checking balance?",                          "balance_query"),
    ("How much money do I have in savings?",                  "balance_query"),
    ("Show me my account balance",                            "balance_query"),
    ("What's my available balance?",                          "balance_query"),
    ("How much is in my account?",                            "balance_query"),
    ("What's my balance",                                     "balance_query"),
    ("Check my balance",                                      "balance_query"),
    ("What is my current balance",                            "balance_query"),
    ("How much do I have",                                    "balance_query"),
    ("What's in my checking account",                         "balance_query"),
    ("Tell me my savings balance",                            "balance_query"),
    ("What is my account balance right now",                  "balance_query"),
    ("How much money is in my savings",                       "balance_query"),
    ("Show my checking account balance",                      "balance_query"),
    ("What is my credit card balance",                        "balance_query"),
    ("How much credit do I have left",                        "balance_query"),
    ("What's my remaining balance",                           "balance_query"),
    ("Can you check my account balance",                      "balance_query"),
    ("How much money do I have right now",                    "balance_query"),
    ("What is the balance in my savings account",             "balance_query"),
    ("Show me how much I have in checking",                   "balance_query"),
    ("What's the balance on my account",                      "balance_query"),
    ("How much is available in my account",                   "balance_query"),
    ("Tell me my current account balance",                    "balance_query"),
    ("What is my total balance across all accounts",          "balance_query"),

    # ── spending_analysis (35 examples) ──────────────────────────────────────
    ("How much did I spend on dining this month?",            "spending_analysis"),
    ("Show me my spending breakdown",                         "spending_analysis"),
    ("What were my top purchases last month?",                "spending_analysis"),
    ("Analyze my transactions",                               "spending_analysis"),
    ("Where is my money going?",                              "spending_analysis"),
    ("How much did I spend this month",                       "spending_analysis"),
    ("What did I spend money on last week",                   "spending_analysis"),
    ("Show my recent transactions",                           "spending_analysis"),
    ("What are my biggest expenses",                          "spending_analysis"),
    ("How much have I spent on groceries",                    "spending_analysis"),
    ("Show me my transaction history",                        "spending_analysis"),
    ("What did I buy at Amazon",                              "spending_analysis"),
    ("How much did I spend on food",                          "spending_analysis"),
    ("Show spending by category",                             "spending_analysis"),
    ("What were my purchases this week",                      "spending_analysis"),
    ("How much did I spend at restaurants",                   "spending_analysis"),
    ("Show me all my charges this month",                     "spending_analysis"),
    ("What did I spend on entertainment",                     "spending_analysis"),
    ("How much went to subscriptions",                        "spending_analysis"),
    ("Show my spending trends",                               "spending_analysis"),
    ("What were my largest transactions",                     "spending_analysis"),
    ("How much did I spend on gas",                           "spending_analysis"),
    ("Show me my purchases from last month",                  "spending_analysis"),
    ("Analyze my spending habits",                            "spending_analysis"),
    ("Where did most of my money go this month",              "spending_analysis"),
    ("analyse my spending",                                   "spending_analysis"),
    ("analyse my transactions",                               "spending_analysis"),
    ("analyse where my money goes",                           "spending_analysis"),
    ("spending analysis for this month",                      "spending_analysis"),
    ("analyse my expenses",                                   "spending_analysis"),
    ("break down my spending",                                "spending_analysis"),
    ("what am I spending money on",                           "spending_analysis"),
    ("spending report",                                       "spending_analysis"),
    ("monthly spending analysis",                             "spending_analysis"),
    ("analyse my budget",                                     "spending_analysis"),

    # ── transfer_request (25 examples) ───────────────────────────────────────
    ("Transfer $500 to my savings account",                   "transfer_request"),
    ("Move money to savings",                                 "transfer_request"),
    ("Send $200 to John via Zelle",                           "transfer_request"),
    ("I want to move $1000",                                  "transfer_request"),
    ("Pay my rent from checking",                             "transfer_request"),
    ("Transfer money between accounts",                       "transfer_request"),
    ("Send money to my friend",                               "transfer_request"),
    ("Move $500 from checking to savings",                    "transfer_request"),
    ("I need to transfer funds",                              "transfer_request"),
    ("Send a payment to Sarah",                               "transfer_request"),
    ("Zelle someone $100",                                    "transfer_request"),
    ("Make a transfer to my other account",                   "transfer_request"),
    ("Send $50 to my roommate",                               "transfer_request"),
    ("Move funds to my savings",                              "transfer_request"),
    ("Transfer $250 to checking",                             "transfer_request"),
    ("I want to send money to my mom",                        "transfer_request"),
    ("Wire money to another account",                         "transfer_request"),
    ("Send $300 via Zelle to Mike",                           "transfer_request"),
    ("Move $1000 to my savings account",                      "transfer_request"),
    ("Transfer to external account",                          "transfer_request"),
    ("I need to send money",                                  "transfer_request"),
    ("Pay someone with Zelle",                                "transfer_request"),
    ("Move money from savings to checking",                   "transfer_request"),
    ("Send funds to another bank",                            "transfer_request"),
    ("Transfer $750 to my investment account",                "transfer_request"),

    # ── fraud_report (25 examples) ────────────────────────────────────────────
    ("I didn't make this transaction",                        "fraud_report"),
    ("There's a charge I don't recognize",                    "fraud_report"),
    ("Someone used my card without permission",               "fraud_report"),
    ("Report a fraudulent charge",                            "fraud_report"),
    ("Dispute this transaction",                              "fraud_report"),
    ("This charge looks suspicious",                          "fraud_report"),
    ("I was charged twice for the same thing",                "fraud_report"),
    ("There's an unauthorized charge on my account",          "fraud_report"),
    ("I think my card was stolen",                            "fraud_report"),
    ("Someone made a purchase I didn't authorize",            "fraud_report"),
    ("This transaction is fraudulent",                        "fraud_report"),
    ("I need to dispute a charge",                            "fraud_report"),
    ("There's a charge from a merchant I never visited",      "fraud_report"),
    ("My card was used without my knowledge",                 "fraud_report"),
    ("I see a transaction I don't recognize",                 "fraud_report"),
    ("This purchase was not made by me",                      "fraud_report"),
    ("Flag this transaction as fraud",                        "fraud_report"),
    ("I want to report identity theft",                       "fraud_report"),
    ("Someone stole my credit card information",              "fraud_report"),
    ("There's a suspicious charge on my statement",           "fraud_report"),
    ("Cancel this unauthorized transaction",                  "fraud_report"),
    ("I never made this purchase",                            "fraud_report"),
    ("My account shows a charge I didn't make",               "fraud_report"),
    ("Report unauthorized account access",                    "fraud_report"),
    ("I need to file a fraud dispute",                        "fraud_report"),

    # ── investment_query (25 examples) ───────────────────────────────────────
    ("How is my portfolio doing?",                            "investment_query"),
    ("Show my investment performance",                        "investment_query"),
    ("What are my stock holdings worth?",                     "investment_query"),
    ("Check my J.P. Morgan account",                          "investment_query"),
    ("Portfolio rebalancing suggestions",                     "investment_query"),
    ("What is my portfolio value",                            "investment_query"),
    ("How are my investments performing",                     "investment_query"),
    ("Show my stock portfolio",                               "investment_query"),
    ("What stocks do I own",                                  "investment_query"),
    ("How much have I made on investments",                   "investment_query"),
    ("What is my investment return",                          "investment_query"),
    ("Show me my retirement account",                         "investment_query"),
    ("How is my 401k doing",                                  "investment_query"),
    ("What are my mutual funds worth",                        "investment_query"),
    ("Should I rebalance my portfolio",                       "investment_query"),
    ("What is my net worth",                                  "investment_query"),
    ("Show my asset allocation",                              "investment_query"),
    ("How are my ETFs performing",                            "investment_query"),
    ("What dividends did I receive",                          "investment_query"),
    ("Show my investment gains and losses",                   "investment_query"),
    ("How much is in my brokerage account",                   "investment_query"),
    ("What are my top performing investments",                "investment_query"),
    ("Show me my IRA balance",                                "investment_query"),
    ("How has my portfolio changed this month",               "investment_query"),
    ("What is my investment account balance",                 "investment_query"),
    ("show my portfolio performance",                         "investment_query"),
    ("review my investment returns",                          "investment_query"),  # ← not "analyse"
    ("how are my investments doing",                          "investment_query"),
    ("J.P. Morgan portfolio value",                           "investment_query"),
    ("stock performance this month",                          "investment_query"),

    # ── savings_advice (25 examples) ─────────────────────────────────────────
    ("How can I save more money?",                            "savings_advice"),
    ("What's the best savings rate?",                         "savings_advice"),
    ("Should I open a high-yield savings account?",           "savings_advice"),
    ("Help me save for a house",                              "savings_advice"),
    ("Compare savings options",                               "savings_advice"),
    ("What is the current APY on savings",                    "savings_advice"),
    ("How much should I save each month",                     "savings_advice"),
    ("What's the best way to grow my savings",                "savings_advice"),
    ("Should I move money to a high yield account",           "savings_advice"),
    ("How can I earn more interest",                          "savings_advice"),
    ("What savings account has the best rate",                "savings_advice"),
    ("Help me build an emergency fund",                       "savings_advice"),
    ("How do I save for retirement",                          "savings_advice"),
    ("What CD rates are available",                           "savings_advice"),
    ("Is a money market account good",                        "savings_advice"),
    ("How can I save for my kids college",                    "savings_advice"),
    ("What are my savings options",                           "savings_advice"),
    ("Help me create a savings plan",                         "savings_advice"),
    ("What interest rate will I earn",                        "savings_advice"),
    ("Should I put money in a CD",                            "savings_advice"),
    ("How can I maximize my savings interest",                "savings_advice"),
    ("What is the difference between savings accounts",       "savings_advice"),
    ("Help me find a better savings rate",                    "savings_advice"),
    ("How much interest am I earning on savings",             "savings_advice"),
    ("What is the APY on my savings account",                 "savings_advice"),

    # ── bill_payment (25 examples) ────────────────────────────────────────────
    ("Pay my credit card bill",                               "bill_payment"),
    ("Set up autopay for utilities",                          "bill_payment"),
    ("Pay my Chase credit card",                              "bill_payment"),
    ("Schedule a bill payment",                               "bill_payment"),
    ("Pay my minimum balance",                                "bill_payment"),
    ("When is my credit card due",                            "bill_payment"),
    ("Pay my electric bill",                                  "bill_payment"),
    ("Set up automatic payments",                             "bill_payment"),
    ("How do I pay my bill",                                  "bill_payment"),
    ("Pay my statement balance",                              "bill_payment"),
    ("Schedule a payment for my credit card",                 "bill_payment"),
    ("Set up autopay",                                        "bill_payment"),
    ("Pay my phone bill",                                     "bill_payment"),
    ("I need to pay my mortgage",                             "bill_payment"),
    ("How much is my minimum payment",                        "bill_payment"),
    ("When is my payment due",                                "bill_payment"),
    ("Pay my car loan",                                       "bill_payment"),
    ("Make a payment on my credit card",                      "bill_payment"),
    ("Schedule my utility payment",                           "bill_payment"),
    ("Pay all my bills this month",                           "bill_payment"),
    ("Set up recurring payment",                              "bill_payment"),
    ("Pay my internet bill",                                  "bill_payment"),
    ("How do I avoid late fees",                              "bill_payment"),
    ("Pay my insurance premium",                              "bill_payment"),
    ("Make a credit card payment",                            "bill_payment"),

    # ── general (25 examples) ─────────────────────────────────────────────────
    ("Hello, how are you?",                                   "general"),
    ("What can you help me with?",                            "general"),
    ("Tell me about Chase benefits",                          "general"),
    ("How do I contact support?",                             "general"),
    ("What is Chase Sapphire?",                               "general"),
    ("Hi there",                                              "general"),
    ("What features do you have",                             "general"),
    ("Help me understand my account",                         "general"),
    ("What is Chase doing about security",                    "general"),
    ("Tell me about Chase rewards",                           "general"),
    ("How do I reset my password",                            "general"),
    ("What is the Chase app",                                 "general"),
    ("Tell me about your AI features",                        "general"),
    ("What can this app do",                                  "general"),
    ("How does Chase AI work",                                "general"),
    ("What are Chase rewards points worth",                   "general"),
    ("How do I update my profile",                            "general"),
    ("What is a routing number",                              "general"),
    ("Tell me about Chase student accounts",                  "general"),
    ("What is Chase Sapphire Reserve",                        "general"),
    ("How do I add a new account",                            "general"),
    ("What banks does Chase work with",                       "general"),
    ("Tell me about Chase travel insurance",                  "general"),
    ("What is the Chase mobile app",                          "general"),
    ("How do I find a Chase ATM",                             "general"),
]

def main():
    texts = [t for t, _ in training]
    labels = [y for _, y in training]

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b"   # keep it simple + predictable
    )

    X = vec.fit_transform(texts)

    clf = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto")
    clf.fit(X, labels)

    os.makedirs("artifacts", exist_ok=True)

    # Export vocabulary + idf
    vocab = vec.vocabulary_  # token -> column index
    idf = vec.idf_.astype(np.float32).tolist()

    # Export classifier
    classes = clf.classes_.tolist()
    # coef_ shape: [n_classes, n_features]
    W = clf.coef_.astype(np.float32).tolist()
    b = clf.intercept_.astype(np.float32).tolist()

    # Other vectorizer settings Swift must match
    export = {
        "classes": classes,
        "vocab": vocab,
        "idf": idf,
        "ngram_range": [1, 2],
        "lowercase": True,
        "strip_accents_unicode": True,
        "token_pattern": r"(?u)\b\w+\b",
        "W": W,
        "b": b
    }

    with open("artifacts/intent_tfidf_lr.json", "w") as f:
        json.dump(export, f)

    print("Saved artifacts/intent_tfidf_lr.json")
    print(f"Num features: {len(idf)}, Num classes: {len(classes)}")

if __name__ == "__main__":
    main()
