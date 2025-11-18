import pandas as pd
import re
import glob
import os
import time
from spellchecker import SpellChecker
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

# It's more efficient to initialize these once per process
# We will pass them using an initializer function
spell_obj_global = None
email_re_global = None
phone_re_global = None
url_re_global = None

def init_worker(spell_obj, email_re, phone_re, url_re):
    """Initializer for multiprocessing workers to set global objects."""
    global spell_obj_global, email_re_global, phone_re_global, url_re_global
    spell_obj_global = spell_obj
    email_re_global = email_re
    phone_re_global = phone_re
    url_re_global = url_re

def process_chunk(df_chunk, text_col):
    """
    Cleans and spell-checks a chunk of the DataFrame.
    Each process running this function will have its own local cache.
    """
    # Local cache for this process only. No locking, no overhead.
    correction_cache = {}

    def clean_and_correct_word(w):
        """Helper to correct a single word using the local cache."""
        if w in correction_cache:
            return correction_cache[w]
        
        if w not in spell_obj_global:
            fixed = spell_obj_global.correction(w)
            correction = fixed if fixed else w
            correction_cache[w] = correction
            return correction
        else:
            correction_cache[w] = w
            return w

    def clean_text_series(s):
        """Cleans a single text entry (a string)."""
        if pd.isna(s): return ""
        t = str(s)
        
        t = email_re_global.sub(' <EMAIL> ', t)
        t = phone_re_global.sub(' <PHONE> ', t)
        t = url_re_global.sub(' <URL> ', t)
        
        t = re.sub(r'[^A-Za-z0-9\s\+\#\.\&\/\-]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip().lower()
        
        words = t.split()
        corrected = [clean_and_correct_word(w) for w in words]
        
        # Remove consecutive duplicates
        final = [corrected[i] for i in range(len(corrected)) if i == 0 or corrected[i] != corrected[i-1]]
        return ' '.join(final)

    # Apply the cleaning function to the entire chunk (Series)
    return df_chunk[text_col].apply(clean_text_series)


if __name__ == '__main__':
    t0 = time.time()
    # Use a direct path for the CSV file for reliability
    csv_file_path = r"C:\Users\muhammad waseem\OneDrive\Desktop\resume_analysis\resume analysis\resumes_clean_final_fast.csv" # This assumes the CSV is in the same folder as your script/notebook

    # Check if the file exists before trying to load it
    if not os.path.exists(csv_file_path):
        print(f"❌ ERROR: Could not find the file: '{csv_file_path}'")
        print(f"Please make sure the file is in the correct location.")
        # This line is very helpful for debugging as it shows where the script is currently looking
        print(f"Current Working Directory: {os.getcwd()}")
        raise SystemExit

    # Load the CSV file
    df = pd.read_csv(csv_file_path, dtype=str, low_memory=False)

    def detect_text_col(df):
        for c in ("clean_text", "resume", "cv", "content", "description"):
            if c in df.columns: return c
        for c in df.columns:
            if df[c].dtype == 'object': return c
        return df.columns[0]

    col = detect_text_col(df)
    print("📄 Using text column:", col)

    # Compile regexes once
    email_re = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
    phone_re = re.compile(r'\+?\d[\d\-\s\(\)]{6,}\d')
    url_re = re.compile(r'https?://\S+|www\.\S+')

    # Create and load the spell checker object once
    spell_obj = SpellChecker()
    spell_obj.word_frequency.load_words([
        "sql", "python", "java", "excel", "ml", "ai", "r", "c++", "c#", "bsc",
        "msc", "django", "flask", "html", "css", "js", "javascript", "pandas",
        "numpy", "powerbi", "hr", "cv", "api", "data", "etl", "machine", "learning"
    ])

    num_processes = mp.cpu_count()
    print(f"⚙️ Using {num_processes} processes for cleaning.")

    # Split the DataFrame into chunks
    df_chunks = np.array_split(df, num_processes)
    
    # Create a list of tuples (chunk, col_name) to pass to the pool
    job_args = [(chunk, col) for chunk in df_chunks]

    with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(spell_obj, email_re, phone_re, url_re)) as pool:
        # We use starmap to pass multiple arguments to process_chunk
        results_list = list(tqdm(pool.starmap(process_chunk, job_args), total=len(df_chunks)))

    # Concatenate the results from all processes
    cleaned_series = pd.concat(results_list)
    df['clean_text'] = cleaned_series

    out = "resumes_clean_spellcheck_chunked.csv"
    df.to_csv(out, index=False)

    print(f"✅ Cleaned & spell-checked file saved: {out}")
    print(f"⏱ Total processing time: {time.time() - t0:.2f} seconds")
