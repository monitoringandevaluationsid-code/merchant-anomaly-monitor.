# GitHub and Streamlit Cloud — exact upload sequence

## GitHub

1. Open your existing repository `merchant-anomaly-monitor` and select the **Code** tab on the `main` branch.
2. At the root, update `app.py` by selecting the existing file, clicking its pencil icon, replacing the contents and selecting **Commit changes**. Alternatively use **Add file → Upload files** and choose the updated files.
3. Ensure the root also contains `requirements.txt` and the new documentation files.
4. Click the blue `outputs` **folder**. Do NOT click **Create new file** to create a file literally named `Output`.
5. Inside `outputs/`, click **Add file → Upload files** and drag the **12** files from the extracted V6 ZIP's `outputs/` directory. If GitHub indicates that names already exist, replace/update the existing files and commit.
6. Check the two new compressed files appear: `transactions_original_captured.csv.gz` and `transactions_new_captured.csv.gz`. The Original one was built from `Transactions.rar` and independently reconciled with the Original prepared daily/category outputs.
7. Add the `templates/` samples as optional demo material.
8. Confirm root `app.py` / root `requirements.txt` / `outputs/` directory and commit to `main`.

## Streamlit Cloud

1. Open your existing deployed app (do not create a new one).
2. Allow the new GitHub commit to redeploy; if necessary use the application management menu to reboot/redeploy.
3. Test **Original → Rachel Sheppard → 13 February 2025**: sales USD 415.32, expected about USD 5,399.13 and same-date category analysis.
4. Test **Original → Transaction Explorer**: filters should display Captured transaction rows from the new compressed file.
5. Test **New → Transaction Explorer** similarly.
6. Test **Upload & Detect** with `templates/sample_transactions_readable.csv`; ensure no extra lookups are required, then check predictions and ZIP download. For coded files, supply all relevant lookup files.

## Operational cautions

- **Do not** upload `Transactions.csv` (31 MB raw), `Transactions.rar` or the original Excel workbook into `outputs/`.
- The original Excel workbook is 94 MB. It documents the preliminary exploration but is not required to run Streamlit.
- This release includes the previously evaluated model prediction files unchanged, and adds the Original Captured transaction extract.
- Aggregate flagged-day alerts and 90-day merchant classifications answer different questions.
- New uploaded data are analysed with **unsupervised screening**, and have no externally confirmed fraud labels.
