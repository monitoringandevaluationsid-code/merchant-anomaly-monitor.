# GitHub and Streamlit steps

1. Download and extract the accompanying ZIP. Open the inner `merchant_seamless_github` folder.
2. Test locally: install requirements, then `streamlit run app.py`.
3. Open github.com/new and create a repository `merchant-anomaly-monitor`.
4. Use Add file > Upload files to upload **the contents** of this project folder, including the `outputs` and `.streamlit` directories. GitHub web upload may not include hidden `.streamlit` automatically; create its config via GitHub editor if needed.
5. Commit changes to `main`.
6. Visit share.streamlit.io, sign in, Create app > Existing repository. Select repository / `main` / entrypoint `app.py`. Deploy.
7. Confirm Original portfolio / Isolation Forest, 105 merchants, 11 model flags, 47 daily markers, category drill down, downloadable CSV.
8. To unlock Original RF/GB/LR and New portfolio, locally run the batch pipeline with all eight raw input CSVs to `generated_outputs`; verify prediction names and evaluator consistency, then upload only the validated prepared outputs into `outputs`.
9. GitHub's web UI limits files to 25 MB; the bundled prepared files are all below this limit. No raw 31 MB transaction file is necessary for hosted dashboard.
10. For the assignment report, distinguish in-sample supervised labels from independently evaluated new data.
