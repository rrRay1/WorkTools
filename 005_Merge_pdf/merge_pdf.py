# Author: Rui Ren
# Copyright (c) 2025 CONTINENTAL AUTOMOTIVE. All rights reserved.
# Description: Merge all PDF files in the current directory into a single PDF, sorted by filename.
# Version: 1.0
# Date: 2025-06-05

import os
from PyPDF2 import PdfMerger

def merge_all_pdfs_in_dir(output_name="merged.pdf"):
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    pdf_files.sort()
    if not pdf_files:
        print("No PDF files found in the current directory.")
        return
    merger = PdfMerger()
    for pdf in pdf_files:
        merger.append(pdf)
        print(f"Added: {pdf}")
    merger.write(output_name)
    merger.close()
    print(f"Merged, output: {output_name}")

if __name__ == "__main__":
    merge_all_pdfs_in_dir()
