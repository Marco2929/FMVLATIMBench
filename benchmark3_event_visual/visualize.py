changes:
    - check only those with ground truth != None
    - distance 0 => ignore from average / mean
    - calculate confusion, as classification is basically if they said None or gave a bbox