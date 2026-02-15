#!/usr/bin/env python3
"""
Gemini-Enhanced Financial Statement Extractor
Uses Google Gemini API to post-process and improve extraction accuracy
Version: 1.0.1
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
from universal_financial_extractor import UniversalFinancialExtractor

try:
    import google.genai as genai
except ImportError:
    print("❌ Google Generative AI library not installed")
    print("Run: pip install google-genai")
    sys.exit(1)

class GeminiEnhancedExtractor(UniversalFinancialExtractor):
    """Enhanced extractor with Gemini-based post-processing for better accuracy"""
    
    def __init__(self, pdf_path: str, use_llm: bool = True, api_key: str = None):
        super().__init__(pdf_path)
        self.use_llm = use_llm
        self.model = None
        
        if use_llm:
            # Initialize Gemini client
            api_key = api_key or os.getenv('GEMINI_API_KEY')
            if api_key:
                try:
                    self.client = genai.Client(api_key=api_key)
                    self.model_name = 'gemini-1.5-flash'
                    print("✓ LLM enhancement enabled (Gemini 1.5 Flash)")
                except Exception as e:
                    print(f"⚠ Failed to initialize Gemini: {e}")
                    print("  LLM enhancement disabled")
                    self.use_llm = False
                    self.client = None
            else:
                print("⚠ GEMINI_API_KEY not found - LLM enhancement disabled")
                self.use_llm = False
                self.client = None
    
    def extract(self, output_path: str = None) -> pd.DataFrame:
        """Extract with LLM enhancement"""
        
        # First, do normal extraction
        df = super().extract(output_path=None)  # Don't save yet
        
        if df.empty:
            return df
        
        # Apply LLM post-processing if enabled
        if self.use_llm and self.client:
            print("\n[LLM] Enhancing extracted data with Gemini...")
            df = self._llm_enhance_table(df)
        
        # Now save the enhanced version
        if output_path:
            print(f"\n[5/7] Creating Excel file...")
            self.create_excel(df, output_path)
        
        return df
    
    def _llm_enhance_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use Gemini to clean and structure the table"""
        
        if df.empty or len(df) == 0:
            return df
        
        # Convert DataFrame to JSON for LLM
        table_preview = df.head(50).to_dict(orient='records')
        
        prompt = f"""You are a financial data cleaning expert. I have extracted financial statement data from a PDF using OCR, but it has some quality issues:

1. Column headers may be incomplete or merged
2. Line item labels may be split across rows or have OCR errors
3. Numbers may be formatted incorrectly or have OCR mistakes
4. Some rows may be empty or contain garbage

Here is the extracted table (first 50 rows):

{json.dumps(table_preview, indent=2)}

Please help me:
1. Clean and standardize column headers (e.g., identify fiscal periods like "FY 25", "Q3 2025", etc.)
2. Merge split line items (e.g., "Revenue from" + "operations" → "Revenue from operations")
3. Fix obvious OCR errors in line item names
4. Identify and remove rows that are clearly headers, separators, or garbage
5. Ensure numeric columns contain only numbers or None

Return ONLY a JSON object with this structure:
{{
  "cleaned_headers": ["Column 1 name", "Column 2 name", ...],
  "rows_to_remove": [list of row indices to delete],
  "line_item_fixes": {{"old_name": "new_name", ...}},
  "suggestions": "Any other observations or issues"
}}

Keep all valid financial line items. Only remove obvious non-data rows.
"""

        try:
            # Use the correct API method for google-genai
            response = self.client.models.generate_content(
                model='models/gemini-1.5-flash',  # Full model path required
                contents=prompt,
                config={
                    'temperature': 0.1,
                    'max_output_tokens': 2000,
                }
            )
            
            # Parse Gemini response
            response_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            llm_output = json.loads(response_text)
            
            # Apply LLM suggestions
            print(f"  ✓ Gemini analyzed {len(df)} rows")
            
            # 1. Update column headers
            if 'cleaned_headers' in llm_output and llm_output['cleaned_headers']:
                new_headers = llm_output['cleaned_headers']
                if len(new_headers) == len(df.columns):
                    # Handle duplicate column names by adding suffixes
                    seen = {}
                    unique_headers = []
                    for header in new_headers:
                        if header in seen:
                            seen[header] += 1
                            unique_headers.append(f"{header}_{seen[header]}")
                        else:
                            seen[header] = 0
                            unique_headers.append(header)
                    df.columns = unique_headers
                    print(f"  ✓ Updated column headers: {unique_headers}")
            
            # 2. Remove bad rows
            if 'rows_to_remove' in llm_output and llm_output['rows_to_remove']:
                rows_to_remove = [idx for idx in llm_output['rows_to_remove'] if idx < len(df)]
                if rows_to_remove:
                    df = df.drop(rows_to_remove).reset_index(drop=True)
                    print(f"  ✓ Removed {len(rows_to_remove)} invalid rows")
            
            # 3. Fix line item names
            if 'line_item_fixes' in llm_output and llm_output['line_item_fixes']:
                fixes = llm_output['line_item_fixes']
                # Use iloc to avoid issues with duplicate column names
                for old, new in fixes.items():
                    df.iloc[:, 0] = df.iloc[:, 0].replace(old, new)
                print(f"  ✓ Fixed {len(fixes)} line item names")
            
            # 4. Show suggestions
            if 'suggestions' in llm_output and llm_output['suggestions']:
                print(f"  💡 Gemini suggestions: {llm_output['suggestions']}")
            
            return df
            
        except Exception as e:
            print(f"  ⚠ LLM enhancement failed: {str(e)}")
            print(f"  → Continuing with original extraction")
            return df
    
    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced post-processing"""
        
        # First apply parent post-processing
        df = super()._post_process(df)
        
        # Additional cleaning for merged text in first column
        if len(df.columns) > 0 and len(df) > 0:
            # Use iloc to access first column by position (not by name)
            # This avoids issues when column name is empty string and duplicated
            
            # Common patterns to clean
            import re
            patterns = [
                # Remove leading numbers (like "1.", "2.", etc.)
                (r'^\d+\.?\s+', ''),
                # Remove multiple spaces
                (r'\s+', ' '),
                # Fix common OCR errors
                (r'\bfrom\s+operations?\b', 'from operations'),
                (r'\bfor\s+the\s+period\b', 'for the period'),
                (r'\bTotal\s+income\b', 'Total income'),
            ]
            
            # Apply patterns to each value in the first column
            cleaned_values = []
            for value in df.iloc[:, 0]:
                if pd.isna(value) or value == '':
                    cleaned_values.append(value)
                else:
                    text = str(value)
                    for pattern, replacement in patterns:
                        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                    cleaned_values.append(text.strip())
            
            # Assign back to first column using iloc
            df.iloc[:, 0] = cleaned_values
        
        return df


def main():
    """Main function"""
    
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Default test files
        test_files = [
            "Dabur Quaterly Financial Statements.pdf",
            "Tata Motors Quarterly Financial Statements.pdf"
        ]
        
        pdf_path = None
        for f in test_files:
            if Path(f).exists():
                pdf_path = f
                break
        
        if not pdf_path:
            print("Usage: python gemini_enhanced_extractor.py <pdf_file>")
            print("\nOr place one of these files in the current directory:")
            for f in test_files:
                print(f"  - {f}")
            return
    
    # Check for API key
    use_llm = bool(os.getenv('GEMINI_API_KEY'))
    
    if not use_llm:
        print("=" * 80)
        print("⚠ GEMINI_API_KEY not set - LLM enhancement will be disabled")
        print("To enable LLM enhancement:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        print("=" * 80)
        print()
    
    # Extract with LLM enhancement
    extractor = GeminiEnhancedExtractor(pdf_path, use_llm=use_llm)
    
    # Generate output filename
    pdf_name = Path(pdf_path).stem
    suffix = "_gemini_enhanced" if use_llm else "_extracted"
    output_path = f"output/{pdf_name}{suffix}.xlsx"
    
    # Run extraction
    df = extractor.extract(output_path)
    
    # Display
    if not df.empty:
        print("\n" + "="*80)
        print("EXTRACTED DATA PREVIEW")
        print("="*80)
        print(df.to_string(index=False, max_rows=30))
        print("\n" + "="*80)
    
    return df


if __name__ == "__main__":
    main()
