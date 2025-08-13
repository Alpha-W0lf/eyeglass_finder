# Google Gemini API Notes

**Version:** 1.1
**Date:** August 12, 2025
**Status:** Verified

This document summarizes key details about the Google Gemini API, focusing on the free tier models and limits relevant to our content creation projects. This information has been verified against official Google AI documentation as of the date above.

---

### 1. Available Models & Suitability

The project has access to several models under the free tier, each suited for different tasks.

| Model                   | Key Characteristics & Use Cases                                                                                              | Input/Output Token Limits |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| **`Gemini 2.5 Pro`**      | Most capable model. Best for high-quality reasoning, complex instruction, and primary content generation (script writing, prompt authoring). | 1M input / 65k output     |
| **`Gemini 2.5 Flash`**    | Faster and less expensive. Ideal for high-volume, moderate-complexity tasks (e.g., initial research synthesis, advisory reviews).    | 1M input / 65k output     |
| **`Gemini 2.5 Flash-Lite`** | Fastest and most cost-effective. Best for very high-volume, low-complexity tasks (e.g., simple data extraction, chunking validation). | 1M input / 65k output     |
| **`text-embedding-004`**  | Generates vector embeddings for text. Essential for any future RAG architecture or semantic search capabilities.                 | 8192 input                |

---

### 2. Free Tier Rate Limits

Rate limits are applied **per project**. The orchestrator must handle these gracefully via queuing, exponential backoff, and careful job scheduling.

**A. Standard ("Interactive") API Calls**

| Model                   | Requests per Minute (RPM) | Tokens per Minute (TPM) | Requests per Day (RPD) |
| ----------------------- | ------------------------- | ----------------------- | ---------------------- |
| **`Gemini 2.5 Pro`**      | 5                         | 250,000                 | 100                    |
| **`Gemini 2.5 Flash`**    | 10                        | 250,000                 | 250                    |
| **`Gemini 2.5 Flash-Lite`** | 15                        | 250,000                 | 1,000                  |
| **`text-embedding-004`**  | 1,500                     | 3,000,000               | -                      |

**B. Batch API Calls**

For non-interactive bulk processing, batch mode offers different limits:
*   **Concurrent Batch Requests:** 100
*   **Input File Size Limit:** 2GB
*   **File Storage Limit:** 20GB

**Implications for Our Project:**
*   The low RPM for `Gemini 2.5 Pro` is the most significant constraint. High-quality generation tasks must be carefully sequenced.
*   `Flash` and `Flash-Lite` models provide much higher request throughput, making them suitable for the "many small tasks" pattern (e.g., reviewing individual script chunks).
*   The embedding model has very generous limits, which will not be a blocker for implementing RAG.

---

### 3. Data Usage & Privacy Policy (Crucial Consideration)

*   **Free Tier (Unpaid Services):**
    *   Google **may use submitted content** (prompts, files, responses) to improve their products.
    *   Human reviewers may access this data.
    *   **Official Policy:** "Do not submit sensitive, confidential, or personal information to the Unpaid Services."
    *   **Our Decision:** Acceptable for our public-facing content creation, as no sensitive PII is involved.

*   **Paid Tier:**
    *   Requires a linked Google Cloud Billing account.
    *   Google **does not** use customer content to improve their products.
    *   Rate limits are significantly higher.

---

### 4. Other Key Information

*   **Authentication:** Via API keys from Google AI Studio.
*   **Context Window:** The 1M token context window for Pro/Flash models is a major advantage, allowing agents to analyze entire scripts in a single call, improving conceptual understanding.
*   **Image Generation (`Imagen 4`):** Access via the Gemini API is a **paid service and is NOT included in the standard free tier.**
    *   **Architectural Decision:** This is **NOT VIABLE** for our project. We must retain the plan to use browser automation for image generation.

---

### 5. Verified Information Source

*   **Primary Source:** [Google AI for Developers - Quotas](https://ai.google.dev/gemini-api/docs/quota)
*   **Pricing/Tier Info:** [Google AI for Developers - Pricing](https://ai.google.dev/gemini-api/docs/pricing)
*   **Models Info:** [Google AI for Developers - Models](https://ai.google.dev/gemini-api/docs/models)
(Links are as of August 2025)
