#!/usr/bin/env python3
"""
Research Portal - Main Application
Upload financial PDFs and extract to Excel automatically
"""

import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
import pandas as pd
import time

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_financial_extractor import UniversalFinancialExtractor
from gemini_enhanced_extractor import GeminiEnhancedExtractor


def make_columns_unique(df):
    """Make DataFrame column names unique by adding suffixes to duplicates"""
    if df.empty:
        return df
    
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    for col in cols:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df


# Page config
st.set_page_config(
    page_title="Financial Research Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stDownloadButton button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-header">📊 Financial Research Portal</p>', unsafe_allow_html=True)
    st.markdown("### Extract Financial Statements from PDF to Excel")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/financial-analytics.png", width=150)
        st.markdown("## About")
        st.info("""
        This portal extracts financial statement line items from PDF documents 
        and converts them to structured Excel files for analysis.
        """)
        
        st.markdown("## ⚙️ Settings")
        
        # LLM Enhancement toggle
        use_llm = st.checkbox(
            "Enable AI Enhancement (Gemini)",
            value=False,
            help="Use Google Gemini AI to improve extraction accuracy"
        )
        
        if use_llm:
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=os.getenv('GEMINI_API_KEY', ''),
                help="Enter your Google Gemini API key"
            )
            if gemini_key:
                os.environ['GEMINI_API_KEY'] = gemini_key
                st.success("✓ API key set")
            else:
                st.warning("⚠️ Enter API key to enable AI enhancement")
        
        st.markdown("---")
        
        st.markdown("## Supported Documents")
        st.markdown("""
        - ✅ Annual Reports
        - ✅ Quarterly Statements
        - ✅ Income Statements
        - ✅ P&L Statements
        """)
        
        st.markdown("## Features")
        st.success("""
        - Multi-year extraction
        - Automatic normalization
        - Excel export
        - Missing data handling
        """)
        
        st.markdown("---")
        st.caption("Powered by Tesseract OCR & OpenCV")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "📖 Documentation", "🔍 Examples"])
    
    with tab1:
        st.header("Upload Financial Statement PDF")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload an annual report or financial statement PDF"
            )
        
        with col2:
            st.markdown("### Quick Start")
            st.markdown("""
            1. Upload PDF
            2. Click Extract
            3. Download Excel
            """)
        
        if uploaded_file:
            st.success(f"✓ Uploaded: **{uploaded_file.name}** ({uploaded_file.size/1024:.1f} KB)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                extract_button = st.button("🚀 Extract Data", type="primary", use_container_width=True)
            
            if extract_button:
                with st.spinner("🔄 Processing PDF... Please wait..."):
                    # Save uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status = st.empty()
                        
                        # Extract
                        status.text("📄 Detecting financial pages...")
                        progress_bar.progress(20)
                        
                        # Choose extractor based on LLM setting
                        if use_llm and os.getenv('GEMINI_API_KEY'):
                            extractor = GeminiEnhancedExtractor(tmp_path, use_llm=True)
                            status.text("🤖 Using AI-enhanced extraction...")
                        else:
                            extractor = UniversalFinancialExtractor(tmp_path)
                            status.text("📄 Using standard extraction...")
                        
                        progress_bar.progress(40)
                        
                        status.text("🔍 Extracting table data...")
                        progress_bar.progress(60)
                        
                        # Create output path
                        output_dir = Path("output")
                        output_dir.mkdir(exist_ok=True)
                        output_path = output_dir / f"extracted_{uploaded_file.name.replace('.pdf', '.xlsx')}"
                        
                        status.text("📊 Processing and creating Excel...")
                        progress_bar.progress(80)
                        
                        df = extractor.extract(str(output_path))
                        
                        # Handle duplicate column names for display
                        df = make_columns_unique(df)
                        
                        if df.empty:
                            st.error("❌ No financial data found in the PDF")
                            
                            st.warning("### Possible Reasons:")
                            st.markdown("""
                            1. **PDF doesn't contain a financial statement** (income statement/P&L)
                            2. **PDF is password protected** or corrupted
                            3. **Line items have non-standard names**
                            4. **PDF is image-based** with poor OCR quality
                            """)
                            
                            st.info("### 💡 Try This:")
                            st.markdown("""
                            - Upload an **annual report** or **quarterly financial statement**
                            - Ensure the PDF contains line items like:
                              - Revenue from operations
                              - Employee expenses
                              - Profit before tax
                            - Try the **Tata Motors sample** first to verify system is working
                            """)
                            
                            # Offer to test with sample
                            if st.button("🧪 Test with Tata Motors Sample PDF"):
                                st.info("Testing with sample file...")
                                with st.spinner("Processing sample..."):
                                    if use_llm and os.getenv('GEMINI_API_KEY'):
                                        sample_extractor = GeminiEnhancedExtractor(
                                            "Tata Motors Quarterly Financial Statements.pdf",
                                            use_llm=True
                                        )
                                    else:
                                        sample_extractor = UniversalFinancialExtractor(
                                            "Tata Motors Quarterly Financial Statements.pdf"
                                        )
                                    sample_df = sample_extractor.extract("output/sample_test.xlsx")
                                    
                                    # Handle duplicate column names
                                    sample_df = make_columns_unique(sample_df)
                                    
                                    if not sample_df.empty:
                                        st.success("✅ Sample extraction works! Your uploaded PDF may have a different format.")
                                        st.dataframe(sample_df, use_container_width=True)
                                    else:
                                        st.error("❌ Even sample failed. Please check Tesseract installation.")
                        else:
                            progress_bar.progress(100)
                            status.text("✅ Extraction complete!")
                            
                            time.sleep(0.5)
                            
                            # Success message
                            st.balloons()
                            st.success("🎉 **Extraction completed successfully!**")
                            
                            # Metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Line Items", len(df))
                            with col2:
                                st.metric("Columns", len(df.columns))
                            with col3:
                                years = [col for col in df.columns if col != 'Particulars']
                                st.metric("Years", len(years))
                            with col4:
                                st.metric("Status", "✓ Ready")
                            
                            # Preview
                            st.markdown("### 📊 Data Preview")
                            st.dataframe(df, use_container_width=True, height=400)
                            
                            # Download button
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download Excel File",
                                    data=f,
                                    file_name=output_path.name,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            
                            # Show years detected
                            if years:
                                st.info(f"📅 **Years detected:** {', '.join(years)}")
                    
                    except Exception as e:
                        st.error(f"❌ **Error:** {str(e)}")
                        with st.expander("Show error details"):
                            st.exception(e)
                    
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
        
        else:
            st.info("👆 **Please upload a PDF file to begin**")
            
            # Show example files and test button
            st.markdown("### 📁 Sample Files Available")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if os.path.exists("Tata Motors Quarterly Financial Statements.pdf"):
                    st.markdown("✓ `Tata Motors Quarterly Financial Statements.pdf`")
                if os.path.exists("Financial Statements Examples.xlsx"):
                    st.markdown("✓ `Financial Statements Examples.xlsx` (Expected output format)")
            
            with col2:
                if st.button("🧪 Test Sample", help="Test with Tata Motors PDF"):
                    st.info("Running test extraction...")
                    with st.spinner("Processing..."):
                        try:
                            if use_llm and os.getenv('GEMINI_API_KEY'):
                                test_extractor = GeminiEnhancedExtractor(
                                    "Tata Motors Quarterly Financial Statements.pdf",
                                    use_llm=True
                                )
                            else:
                                test_extractor = UniversalFinancialExtractor(
                                    "Tata Motors Quarterly Financial Statements.pdf"
                                )
                            df = test_extractor.extract("output/sample_demo.xlsx")
                            
                            if not df.empty:
                                st.success("✅ Test successful! System is working correctly.")
                                st.dataframe(df, use_container_width=True)
                                
                                with open("output/sample_demo.xlsx", 'rb') as f:
                                    st.download_button(
                                        "⬇️ Download Sample Output",
                                        data=f,
                                        file_name="tata_motors_sample.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                            else:
                                st.error("Test failed. Check Tesseract installation.")
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    with tab2:
        st.header("📖 How It Works")
        
        st.markdown("""
        ## 🔄 Extraction Pipeline
        
        Our system uses a **3-step process** to extract financial data:
        
        ### Step 1: OCR Text Extraction
        - Converts PDF pages to images (200 DPI)
        - Uses Tesseract OCR to extract text
        - Preserves layout and structure
        
        ### Step 2: Intelligent Parsing
        - **Year Detection**: Automatically finds fiscal years (FY 25, FY 24, etc.)
        - **Line Item Recognition**: Matches text to standard financial terms
        - **Number Extraction**: Handles various formats:
          - Commas: `1,234.56`
          - Negatives: `(1,234)` or `-1,234`
          - Missing: `—` or blank
        
        ### Step 3: Excel Generation
        - Creates formatted Excel file
        - Applies professional styling
        - Organizes by fiscal year
        
        ---
        
        ## 🤔 Design Decisions
        
        ### 1. **How do we find line items?**
        We maintain a dictionary of standard financial terms and their variations:
        - "Revenue from ops" → "Revenue from operations"
        - "Operating Costs" → "Operating expenses"
        
        ### 2. **Different line item names?**
        Our normalization engine maps variations to standard names while keeping originals for reference.
        
        ### 3. **Missing line items?**
        We leave cells **empty** rather than guessing - this allows analysts to spot gaps easily.
        
        ### 4. **Numeric value extraction reliability**
        - Extract only when confidence is high
        - Validate against expected ranges
        - **No hallucination**: If unclear, leave blank
        
        ### 5. **Currency and units detection**
        - Detect from document text (e.g., "Rs in Crores")
        - Display prominently in metadata
        - Default to INR Crores if uncertain
        
        ### 6. **Multiple years handling**
        - Extract ALL years found (up to 6)
        - Create separate column for each year
        - Sort chronologically (newest first)
        
        ### 7. **Ambiguous data presentation**
        - Empty cells = truly missing data
        - `—` or `N/A` = not applicable
        - Allows easy review and correction by analysts
        
        ---
        
        ## 📋 Supported Line Items
        
        | Standard Name | Variations Recognized |
        |--------------|----------------------|
        | Revenue from operations | revenue from ops, operating revenue, sales |
        | Other income | other sources, non-operating income |
        | Cost of materials consumed | raw material cost, material consumed |
        | Employee benefits expense | employee cost, staff costs, salaries |
        | Finance costs | interest expense, borrowing costs |
        | Depreciation | depreciation and amortisation |
        | Profit before tax | PBT, profit before taxation |
        | Tax expense | income tax, current tax |
        | Profit for the year | PAT, net profit |
        
        """)
    
    with tab3:
        st.header("🔍 Example Output")
        
        st.markdown("### Sample Extraction Result")
        
        # Show expected format
        sample_data = {
            'Particulars': [
                'Revenue from operations',
                'Other income',
                'Total Revenue',
                'Cost of materials consumed',
                'Employee benefits expense',
                'Finance costs',
                'Depreciation',
                'Other expenses',
                'Profit before tax',
                'Tax expense',
                'Profit for the year'
            ],
            'FY 25': [204813.0, 1212.0, 206025.0, 82937.43, 18850.26, 4736.0, 6295.08, 21187.0, 12070.0, 2948.0, 8556.0],
            'FY 24': [163210.0, 793.6, 164003.6, 70264.61, 14465.87, 2784.0, 5502.85, 24184.0, 7631.0, 2120.0, 5485.0],
            'FY 23': [133905.0, 388.49, 134293.49, 64170.92, 12166.42, 2174.0, 4792.2, None, 9454.0, None, 7673.0]
        }
        
        df_sample = pd.DataFrame(sample_data)
        st.dataframe(df_sample, use_container_width=True)
        
        st.markdown("""
        ### ✨ Key Features
        
        1. **Organized Columns**: Each fiscal year gets its own column
        2. **Standard Names**: All line items normalized to standard terminology
        3. **Professional Formatting**: Numbers formatted with commas and decimals
        4. **Missing Data**: Shown as empty cells (see FY 23 "Other expenses")
        5. **Ready for Analysis**: Can immediately start calculations
        """)
        
        # Show actual extracted file if exists
        if os.path.exists("output/final_extraction.xlsx"):
            st.markdown("---")
            st.markdown("### 📊 Actual Extraction (Tata Motors)")
            df_actual = pd.read_excel("output/final_extraction.xlsx")
            st.dataframe(df_actual, use_container_width=True)
            
            with open("output/final_extraction.xlsx", 'rb') as f:
                st.download_button(
                    label="⬇️ Download Sample Output",
                    data=f,
                    file_name="tata_motors_sample.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Financial Research Portal v1.0 | Built with ❤️ using Streamlit & Tesseract OCR"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
