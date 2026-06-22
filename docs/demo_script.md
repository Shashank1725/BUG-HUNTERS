# Dell FutureMinds AI Hackathon — 2-Minute Demo Script

## Part 1: Main Live Demo Script (Total: ~1m 30s)

| Time | Action on Screen | What to Say (Word-for-Word) | What to Highlight/Point Out |
| :--- | :--- | :--- | :--- |
| **0:00 - 0:20** | Show the Streamlit UI homepage. Drag and drop `quarterly_report.pdf` into the uploader. | "Hello judges. Traditional RAG systems fail on complex documents because they treat layout hierarchy, tables, and captions as isolated chunks. Today, we present BUG-HUNTERS: a layout-aware multi-modal semantic engine. I will upload our Quarterly Business Report." | Point to the **File Uploader** showing the file `quarterly_report.pdf` has loaded. |
| **0:20 - 0:45** | Click the **"Ask"** button with the default question: *"Which region contributed the most revenue?"* | "When I click 'Ask', the document is processed dynamically. We extract elements, construct a layout-relationship graph, ingest into an in-memory vector store, expand retrieved nodes via layout links, and generate a final verified answer using the Groq LLaMA 3 model." | Highlight the green checkmarks appearing on the **Pipeline Progress** dashboard (`✓ Parsing` → `✓ Graph Building` → `✓ Retrieval` → `✓ QA Generation`). |
| **0:45 - 1:15** | Scroll down to show the generated **Answer**, **Confidence**, and **Citations**. | "As you can see, the engine successfully retrieved the correct answer: **North America**. More importantly, it gives us a confidence score of **1.0** and lists the exact citation card showing **Element ID quarterly_report_p1_el12** on **Page 1**, including the precise snippet. This provides 100% trace lineage and eliminates LLM hallucinations." | Highlight the **Confidence Score** and point directly to the **Citation Card** containing the snippet. |
| **1:15 - 1:45** | Show the architecture diagram or sidebar controls. | "As the Integration Engineer, my role was to align the schema contracts between our parser, layout-graph model, and retrieval database using an adapter layer, validate the pipeline end-to-end, and build this production-ready Streamlit dashboard. By bridging our layout graph with dense vector search, we ensure that if a query hits a table, the retriever dynamically pulls its describing caption and nearby headers, restoring critical context." | Highlight the **Configuration Sidebar** and mention how the adapter operates seamlessly between the modules. |

---

## Part 2: Backup Script (In case of Groq API, internet, or lag failures)

If the Groq API fails or is too slow, follow these backup steps:

1. **Keep a backup browser tab open:** Run the demo locally using the command-line script beforehand, and keep a pre-loaded browser tab containing the successful response on screen.
2. **Switch tabs immediately if it spins for more than 10 seconds.**
3. **What to Say (Backup Speech):**
   * *"While the live API request is fetching, let me show you the pre-loaded result. In this run, the engine successfully extracted 'North America' directly from the financial table. The graph retrieval ensured that the table caption and the header were linked together in the context window. This proves our integration successfully resolves the context isolation issue of naive RAG."*

---

## Technical Q&A Cheat Sheet (Judge Questions)

* **Q: Why is your graph retriever better than chunking?**
  * *A: Chunking chops text blindly. Our graph models visual and structural links. For example, if a table contains revenue numbers and the caption contains the explanation, a vector search might retrieve only one. Our graph retriever starts at the table and traverses the `caption_of` relationship edge to pull both, ensuring the LLM gets the full context.*
