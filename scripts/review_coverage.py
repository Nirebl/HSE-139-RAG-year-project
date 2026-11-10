"""Aggregate human labels without treating unjudged candidates as irrelevant."""
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
rows=pd.read_csv(ROOT/'reports/coverage_review.csv',keep_default_na=False)
bad=rows[~rows.human_label.astype(str).isin(['','0','1','2'])]
if len(bad):
    raise ValueError('human_label must be blank or 0/1/2')
judged=rows[rows.human_label.astype(str).isin(['0','1','2'])].copy()
if len(judged) and ((judged.reviewer=='') | (judged.reviewed_at=='')).any():
    raise ValueError('Human labels require reviewer and reviewed_at')
if judged.empty:
    print('Ожидает ручной проверки; оценок человека нет. Coverage не подтверждено.')
    sys.exit(0)
judged['human_label']=judged.human_label.astype(int)
coverage=judged.groupby(['query_id','source_type']).agg(reviewed=('human_label','size'),best_label=('human_label','max')).reset_index()
coverage.to_csv(ROOT/'reports/human_coverage.csv',index=False)
print(coverage.to_string(index=False))
print('Показано покрытие среди проверенных кандидатов; непроверенные документы остаются неизвестными.')
