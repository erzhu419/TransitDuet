# V503 Confirmation

The eight small JSON files are extracts of the completed scheduler tasks
t224033 through t224040, in the arm/seed order listed in submission_train.json.
They retain the fields read by the frozen aggregate, including actual
upper/lower update counts, day IDs, common tape IDs and physical results.
Training curves and weights remain in the server run directories.

From the study directory:

    python3 study.py aggregate --reviews confirmation

The frozen joint criterion was not met. Keep the complete paired sample,
including the high-delay swapped run; it is part of the primary estimate.
