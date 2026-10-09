# Stage 1 — Excel exploration and PivotTable evidence

The project began by exploring the **Original** merchant data in Microsoft Excel before building the reproducible Python pipeline. The user's `Merchant_Analysis_Original(3).xlsx` workbook contains eight worksheets: **Transactions**, **Merchant**, **Business**, **Status**, **Checks**, **Daily Sales**, **Merchant View**, and **Anomaly Checks**. Inspection identified **two PivotTable objects** and an embedded chart.

The initial stage used Excel filters, lookup checks and PivotTables for: (1) distinguishing payment status, especially **Captured**, (2) summarising sales by merchant and date, (3) examining merchant-specific time-series behaviour, and (4) reviewing deviations and possible anomalies. The saved Checks sheet reconciles **965,075 Original transactions**, **482,835 Captured transactions** and **USD 158,371,987.40** in Captured sales.

Python then re-performed filtering, lookup joins and aggregation to produce the 9,450 merchant–date observations and 47,250 merchant–category–date observations. The submitted model outputs and external evaluator results were derived from the Python workflow, **not attributed to the Excel PivotTables**.

**Why the workbook is not in this repository:** the Excel source workbook is about **94 MB**, above GitHub's browser upload limit. Keep it as an offline academic evidence artefact; do not upload it into `outputs/`. The GitHub repo instead includes the exact deployment datasets and this description of the original exploratory stage.
