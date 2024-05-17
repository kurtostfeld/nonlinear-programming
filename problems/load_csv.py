import csv
import torch


def load_csv_to_tensor(path: str) -> torch.Tensor:
    matrix_rows = []
    max_width = 0
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for raw_row in reader:
            # print(f"got row: {raw_row}")
            parsed_row = [float(x) for x in raw_row]
            if len(parsed_row) > max_width:
                max_width = len(parsed_row)
            # print(f"parsed_row: {parsed_row}")
            matrix_rows.append(parsed_row)

    if max_width > 1:
        t = torch.tensor(matrix_rows, dtype=torch.double)
    else:
        flattened = [x[0] for x in matrix_rows]
        t = torch.tensor(flattened, dtype=torch.double)

    # print(f"path={path}. tensor shape={t.shape}. contents are...")
    return t