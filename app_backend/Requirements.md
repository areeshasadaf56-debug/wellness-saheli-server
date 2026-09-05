# Wellness Saheli — Requirements Document

## 1. Project Overview
Wellness Saheli is a women's reproductive health mobile application (Flutter) backed by a Python/FastAPI service. It provides menstrual cycle tracking, PCOS risk screening via a trained ML model, contraceptive eligibility checking (WHO MEC-based), an AI check-in companion, and a unified health diary.

## 2. Problem Statement
Women, particularly in under-resourced settings, lack a single accessible tool that combines cycle tracking, PCOS screening, contraceptive guidance, and supportive conversation without requiring a clinic visit for basic informational needs. Existing apps typically cover only one of these areas in isolation.

## 3. Project Objectives
- OBJ-01: Provide accurate menstrual cycle and ovulation tracking.
- OBJ-02: Provide an ML-based PCOS risk screening tool (not diagnostic).
- OBJ-03: Provide contraceptive method eligibility guidance based on WHO Medical Eligibility Criteria.
- OBJ-04: Provide a supportive, non-diagnostic AI check-in with safety handling for crisis situations.
- OBJ-05: Consolidate all user health data into one synced, cross-device diary.

## 4. Target Users
- Primary: Women of reproductive age seeking cycle tracking and reproductive health information.
- Secondary: Users evaluating contraceptive options who want a quick eligibility reference before consulting a provider.

## 5. Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | The system shall allow a user to create an account using name, email, and password. |
| FR-02 | The system shall allow a user to sign in using email and password. |
| FR-03 | The system shall allow a user to reset their password by email. |
| FR-04 | The system shall allow app usage without an account, using a locally generated anonymous device ID. |
| FR-05 | The system shall allow a user to record last period start date, cycle length, and period duration. |
| FR-06 | The system shall calculate and display the current cycle day and phase. |
| FR-07 | The system shall estimate the fertile window and ovulation day from cycle data. |
| FR-08 | The system shall allow logging of daily symptoms and mood. |
| FR-09 | The system shall accept 22 clinical/lifestyle fields and return a PCOS risk prediction with probability. |
| FR-10 | The system shall display which trained model produced a PCOS prediction. |
| FR-11 | The system shall list contraceptive methods with effectiveness data. |
| FR-12 | The system shall accept selected medical conditions and return a safety category (1–4) per contraceptive method. |
| FR-13 | The system shall reject eligibility requests containing unrecognized condition IDs with a specific error. |
| FR-14 | The system shall provide a conversational AI check-in that asks about lifestyle, mood, and reproductive health. |
| FR-15 | The system shall detect crisis-indicating language in check-in messages and return crisis-support resources. |
| FR-16 | The system shall detect described physical medical emergencies separately from psychological crisis. |
| FR-17 | The system shall respond to check-in messages with a fallback message if no AI service key is configured. |
| FR-18 | The system shall persist a user's full health profile (demographics, lifestyle, reproductive history, PCOS history, AI conversation log) server-side, keyed by user ID. |
| FR-19 | The system shall present a unified, chronologically sorted timeline of PCOS results, contraception changes, eligibility checks, and AI conversations. |
| FR-20 | The system shall allow editing of cycle data, display name, and privacy settings. |
| FR-21 | The system shall provide educational content on reproductive anatomy, hormones, menstruation, and endometriosis. |

## 6. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-01 | Passwords shall be hashed (bcrypt) before storage; plain-text passwords shall never be stored or logged. |
| NFR-02 | PCOS prediction shall return a result in under 1 second once the request reaches the backend. |
| NFR-03 | The AI check-in shall degrade gracefully (clear fallback message) if the AI service is unreachable or unconfigured. |
| NFR-04 | All backend input shall be validated, returning specific 4xx errors naming the invalid field rather than unhandled server errors. |
| NFR-05 | The system shall run on Android, iOS, and web from a single Flutter codebase. |
| NFR-06 | The backend shall run on commodity/low-cost hosting (SQLite, no external DB server required). |
| NFR-07 | No API keys or credentials shall be committed to version control. |

## 7. User Roles
- Guest (anonymous, device-ID based)
- Registered User (email/password account)

No admin/clinician role exists in the current system.

## 8. System Features
Cycle Tracking; Ovulation & Fertility; PCOS Detection; Contraceptive Eligibility; AI Check-In; Health Diary; Learn (educational content); Endometriosis Info; Settings & Privacy.

## 9. AI Requirements
- AI-01: The AI check-in shall never issue a diagnosis (physical or mental health).
- AI-02: The AI check-in shall never provide medication dosages or treatment directives.
- AI-03: Crisis-language detection shall be deterministic (keyword-based) and independent of AI model behavior.
- AI-04: The AI provider shall be configurable via environment variable, not hardcoded, to allow provider changes (currently Groq).

## 10. Database Requirements
- DB-01: User accounts (id, name, email, password_hash) shall be stored in a `users` table.
- DB-02: Health profiles shall be stored as a single JSON document per user_id in a `profiles` table (schema-flexible by design — new fields require no migration).
- DB-03: SQLite shall be used as the database engine.

## 11. Security Requirements
- SEC-01: Passwords shall be hashed with bcrypt.
- SEC-02: Sign-in shall return an identical error message for "unknown email" and "wrong password" to prevent user enumeration.
- SEC-03: No secrets (API keys, database files) shall be committed to git.
- **Known gap:** `/profile/{user_id}` is not currently authenticated (no session token is issued or checked). Documented as a required improvement before any real-world deployment.

## 12. Hardware Requirements
- Client: any Android/iOS device or modern web browser capable of running Flutter apps.
- Server: any host capable of running Python 3.10+ (tested on PythonAnywhere free tier).

## 13. Software Requirements
- Frontend: Flutter SDK (stable channel), Dart.
- Backend: Python 3.10+, FastAPI, Uvicorn, SQLite, bcrypt, Groq SDK, scikit-learn, joblib.

## 14. Constraints
- Free-tier hosting (PythonAnywhere) limits disk quota (512 MB) and CPU seconds — influences dependency choices (e.g., avoiding heavy training-only libraries in the deployed environment).
- Free-tier AI provider (Groq) used instead of a paid provider due to credit-balance limitations encountered during development.

## 15. Assumptions
- Users have basic literacy in English (or Urdu, if the language toggle feature is completed).
- Users have internet access when using cloud-backed features (PCOS prediction, chat, eligibility, sync); cycle tracking works offline-first with sync-when-available.

## 16. Out-of-Scope Features
- Clinical diagnosis of any condition (explicitly excluded — the app screens, it does not diagnose).
- Telemedicine / direct provider consultation.
- Payment processing.
- Multi-language support beyond English/Urdu (if completed).

## 17. Acceptance Criteria
- AC-01 (→ FR-01, FR-02): A user can register and subsequently sign in with the same credentials.
- AC-02 (→ FR-09): Submitting a valid 22-field form returns a prediction label, probability, and model name.
- AC-03 (→ FR-12): Submitting selected conditions returns a category (1–4) for each of the 10 listed methods.
- AC-04 (→ FR-15): A message containing crisis-indicating language returns `crisis_concern: true` and real crisis-line resources.
- AC-05 (→ FR-18): Data saved via PUT is retrievable via GET for the same user_id.