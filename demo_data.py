"""Fictional, deliberately small fixtures. Not actual research results."""
DOCUMENTS={
 'demo-retrieval.txt':[
  'Synthetic experiment: Project Cedar uses 180-word passages with 35-word overlap. The index preserves the document name and PDF page position for each passage. This text describes a fictional experiment for testing a document assistant.',
  'In the fictional Cedar evaluation, the test dataset contains 20 questions. Five questions are unanswerable from the documents. Retrieval hit rate means the expected evidence appears among the top four retrieved passages. Human reviewers separately assess answer correctness and whether citations support factual claims.',
  'Cedar uses a local embedding model called nomic-embed-text. Answers are generated locally. Citation-ID validation checks that cited source IDs exist; it does not prove factual correctness. Scanned PDFs require OCR, which the prototype does not implement.'
 ],
 'demo-models.txt':[
  'Synthetic model comparison: Model Birch answered 12 of 15 answerable questions correctly. Model Maple answered 10 of 15 correctly. These invented numbers are demo fixtures, not benchmarks of real models.',
  'In the synthetic comparison, Birch had a median response time of 8 seconds and Maple had a median response time of 5 seconds. Both models were tested on the same corpus and questions. No training data or parameter counts are reported.'
 ]
}
CASES=[
 {'question':'What passage size and overlap does Cedar use?','answerable':True,'expected_answer':'180 words with 35-word overlap.','expected_sources':[{'document':'demo-retrieval.txt','page':1}]},
 {'question':'How many evaluation questions are unanswerable?','answerable':True,'expected_answer':'Five of twenty.','expected_sources':[{'document':'demo-retrieval.txt','page':2}]},
 {'question':'Does checking citation IDs prove factual correctness?','answerable':True,'expected_answer':'No. It only checks that cited IDs exist.','expected_sources':[{'document':'demo-retrieval.txt','page':3}]},
 {'question':'Which fictional model answered more questions correctly?','answerable':True,'expected_answer':'Birch: 12 of 15, compared with Maple: 10 of 15.','expected_sources':[{'document':'demo-models.txt','page':1}]},
 {'question':'What is the median response time of Maple?','answerable':True,'expected_answer':'5 seconds.','expected_sources':[{'document':'demo-models.txt','page':2}]},
 {'question':'What is the parameter count of Birch?','answerable':False,'expected_answer':'Insufficient evidence.','expected_sources':[]}
]
