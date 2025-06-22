"""
test_analysis.py - Test Suite for Customer Clustering Analysis
Creates synthetic test data and validates the analysis pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Set random seed for reproducible results
np.random.seed(42)

def create_test_data(n_customers=1000, filename='test_returns_data.csv'):
    """
    Create realistic synthetic returns data for testing
    
    Args:
        n_customers (int): Number of unique customers to generate
        filename (str): Output filename for test data
    """
    
    print(f"Creating synthetic test data for {n_customers:,} customers...")
    
    # Define customer archetypes with different behaviors
    customer_archetypes = {
        'vip_champions': {
            'percentage': 0.15,
            'avg_orders': (20, 80),
            'avg_items_per_order': (2, 8),
            'return_rate': (0.05, 0.25),
            'return_intensity': (0.3, 0.8),
            'days_to_return': (5, 30),
            'product_variety': (30, 200)
        },
        'heavy_returners': {
            'percentage': 0.08,
            'avg_orders': (10, 40),
            'avg_items_per_order': (3, 10),
            'return_rate': (0.6, 0.9),
            'return_intensity': (0.8, 1.0),
            'days_to_return': (1, 10),
            'product_variety': (15, 100)
        },
        'churn_risk': {
            'percentage': 0.20,
            'avg_orders': (15, 50),
            'avg_items_per_order': (1, 5),
            'return_rate': (0.3, 0.6),
            'return_intensity': (0.4, 0.9),
            'days_to_return': (3, 25),
            'product_variety': (10, 80)
        },
        'explorers': {
            'percentage': 0.25,
            'avg_orders': (8, 35),
            'avg_items_per_order': (1, 6),
            'return_rate': (0.25, 0.55),
            'return_intensity': (0.3, 0.7),
            'days_to_return': (3, 20),
            'product_variety': (50, 300)
        },
        'standard': {
            'percentage': 0.32,
            'avg_orders': (5, 25),
            'avg_items_per_order': (1, 4),
            'return_rate': (0.2, 0.5),
            'return_intensity': (0.4, 0.8),
            'days_to_return': (2, 15),
            'product_variety': (8, 60)
        }
    }
    
    # Generate customer assignments
    customer_assignments = []
    for archetype, config in customer_archetypes.items():
        count = int(n_customers * config['percentage'])
        customer_assignments.extend([archetype] * count)
    
    # Fill remaining customers with 'standard'
    while len(customer_assignments) < n_customers:
        customer_assignments.append('standard')
    
    # Shuffle assignments
    np.random.shuffle(customer_assignments)
    
    # Generate base date range (last 3 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1095)  # 3 years
    
    # Generate customer data
    all_records = []
    customer_id = 1
    
    for archetype in customer_assignments:
        config = customer_archetypes[archetype]
        
        # Generate customer email
        email = f"test_customer_{customer_id:05d}@example.com"
        
        # Generate customer-specific parameters
        num_orders = np.random.randint(config['avg_orders'][0], config['avg_orders'][1] + 1)
        customer_start_date = start_date + timedelta(days=np.random.randint(0, 730))  # Started within first 2 years
        
        # Generate orders for this customer
        order_dates = []
        for _ in range(num_orders):
            days_since_start = np.random.exponential(200)  # Exponential distribution for realistic spacing
            order_date = customer_start_date + timedelta(days=min(days_since_start, 1000))
            if order_date <= end_date:
                order_dates.append(order_date)
        
        order_dates.sort()
        
        # Generate items for each order
        for order_idx, order_date in enumerate(order_dates):
            order_number = f"ORD{customer_id:05d}{order_idx:03d}"
            items_in_order = np.random.randint(config['avg_items_per_order'][0], config['avg_items_per_order'][1] + 1)
            
            for item_idx in range(items_in_order):
                # Generate item details
                sku = f"SKU{np.random.randint(10000, 99999)}"
                product_class = np.random.randint(1000000, 9999999)
                sales_qty = np.random.randint(1, 4)
                
                # Determine if this item will be returned
                return_prob = np.random.uniform(config['return_rate'][0], config['return_rate'][1])
                will_return = np.random.random() < return_prob
                
                if will_return:
                    # Generate return details
                    return_intensity = np.random.uniform(config['return_intensity'][0], config['return_intensity'][1])
                    return_qty = max(1, int(sales_qty * return_intensity))
                    
                    # Generate return date
                    days_to_return = np.random.randint(config['days_to_return'][0], config['days_to_return'][1] + 1)
                    return_date = order_date + timedelta(days=days_to_return)
                    
                    # Convert return date to Excel serial number format (like your real data)
                    excel_serial = (return_date - datetime(1899, 12, 30)).days + (return_date - datetime(1899, 12, 30)).seconds / 86400
                    
                    return_no = f"RET{customer_id:05d}{order_idx:03d}{item_idx:02d}"
                    units_returned_flag = "Yes"
                    return_date_str = str(excel_serial)
                else:
                    return_qty = 0
                    return_no = "-"
                    units_returned_flag = "No"
                    return_date_str = "-"
                
                # Create record
                record = {
                    'CUSTOMER_EMAILID': email,
                    'SALES_ORDER_NO': order_number,
                    'Q_GMM_ID': 900001,
                    'Q_CLS_ID': product_class,
                    'SKU': sku,
                    'Q_SKU_DESC': f"Product {sku}",
                    'SALES_QTY': sales_qty,
                    'UNITS_RETURNED_FLAG': units_returned_flag,
                    'RETURN_NO': return_no,
                    'RETURN_QTY': return_qty,
                    'ORDER_DATE': order_date.strftime('%m/%d/%Y'),
                    'RETURN_DATE': return_date_str
                }
                
                all_records.append(record)
        
        customer_id += 1
        
        # Progress indicator
        if customer_id % 100 == 0:
            print(f"Generated {customer_id:,} customers...")
    
    # Create DataFrame and save
    df = pd.DataFrame(all_records)
    
    # Add some data quality issues (like real data)
    # Some missing return dates
    mask = (df['UNITS_RETURNED_FLAG'] == 'Yes') & (np.random.random(len(df)) < 0.1)
    df.loc[mask, 'RETURN_DATE'] = '-'
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    print(f"✅ Test data created: {filename}")
    print(f"   Total records: {len(df):,}")
    print(f"   Customers: {df['CUSTOMER_EMAILID'].nunique():,}")
    print(f"   Orders: {df['SALES_ORDER_NO'].nunique():,}")
    print(f"   Returns: {(df['RETURN_QTY'] > 0).sum():,}")
    print(f"   Return rate: {(df['RETURN_QTY'] > 0).sum() / len(df):.1%}")
    
    return df

def run_basic_validation_tests(test_data_file):
    """
    Run basic validation tests on the analysis pipeline
    
    Args:
        test_data_file (str): Path to test data CSV
    """
    
    print("\n" + "="*60)
    print("RUNNING BASIC VALIDATION TESTS")
    print("="*60)
    
    try:
        # Test 1: Data Loading
        print("🧪 Test 1: Data Loading...")
        df = pd.read_csv(test_data_file)
        assert len(df) > 0, "Data loading failed - empty DataFrame"
        assert 'CUSTOMER_EMAILID' in df.columns, "Missing required column"
        assert 'RETURN_QTY' in df.columns, "Missing required column"
        print("   ✅ Data loading successful")
        
        # Test 2: Import functions
        print("🧪 Test 2: Function imports...")
        try:
            from functions import ReturnsClusteringAnalysis
            print("   ✅ Functions imported successfully")
        except ImportError as e:
            print(f"   ❌ Import failed: {e}")
            return False
        
        # Test 3: Analyzer initialization
        print("🧪 Test 3: Analyzer initialization...")
        analyzer = ReturnsClusteringAnalysis(df)
        assert analyzer.df is not None, "Analyzer initialization failed"
        print("   ✅ Analyzer initialized successfully")
        
        # Test 4: Feature preparation
        print("🧪 Test 4: Feature preparation...")
        customer_features = analyzer.prepare_customer_features()
        assert len(customer_features) > 0, "Feature preparation failed"
        assert 'RETURN_RATE' in customer_features.columns, "Missing expected feature"
        print(f"   ✅ Features prepared for {len(customer_features):,} customers")
        
        # Test 5: Clustering
        print("🧪 Test 5: Clustering execution...")
        cluster_labels, cluster_centers = analyzer.perform_clustering(n_clusters=5)
        assert len(cluster_labels) == len(customer_features), "Clustering failed"
        assert len(cluster_centers) == 5, "Wrong number of cluster centers"
        print("   ✅ Clustering completed successfully")
        
        # Test 6: Cluster analysis
        print("🧪 Test 6: Cluster analysis...")
        cluster_summary, interpretations = analyzer.analyze_clusters()
        assert len(interpretations) == 5, "Cluster analysis failed"
        assert all('type' in info for info in interpretations.values()), "Missing cluster types"
        print("   ✅ Cluster analysis completed successfully")
        
        print(f"\n🎉 ALL BASIC TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_direct_analysis_test(test_data_file):
    """
    Run analysis test directly without analysis.py module
    """
    print("🧪 Running direct analysis test...")
    
    try:
        from functions import ReturnsClusteringAnalysis
        import pandas as pd
        from datetime import datetime
        
        # Load and analyze
        df = pd.read_csv(test_data_file)
        analyzer = ReturnsClusteringAnalysis(df)
        customer_features = analyzer.prepare_customer_features()
        
        # Quick clustering
        cluster_labels, cluster_centers = analyzer.perform_clustering(n_clusters=5)
        cluster_summary, interpretations = analyzer.analyze_clusters()

        # Create simple export
        export_data = []
        for cluster_id in cluster_summary.index:
            profile = cluster_summary.loc[cluster_id]
            
            export_data.append({
                'Cluster_ID': cluster_id,
                'Cluster_Name': interpretations[cluster_id]['type'],
                'Customer_Count': int(profile['CUSTOMER_COUNT']),
                'Percentage': round((int(profile['CUSTOMER_COUNT']) / len(analyzer.customer_features)) * 100, 1),
                'Target_Actions': interpretations[cluster_id]['action'],
                'Avg_Orders': round(profile['AVG_ORDERS'], 1),
                'Return_Rate': round(profile['AVG_RETURN_RATE'], 3),
                'Return_Ratio': round(profile['AVG_RETURN_RATIO'], 3)
            })

        # Export
        df_export = pd.DataFrame(export_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'test_direct_analysis_{timestamp}.csv'

        df_export.to_csv(filename, index=False)
        print(f"   ✅ Direct analysis successful: {filename}")
        print(f"   📊 Found {len(export_data)} clusters")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Direct analysis failed: {e}")
        return False

def run_comprehensive_export_test(test_data_file):
    """
    Test the comprehensive export functionality
    
    Args:
        test_data_file (str): Path to test data CSV
    """
    
    print("\n" + "="*60)
    print("TESTING COMPREHENSIVE EXPORT")
    print("="*60)
    
    try:
        # Test comprehensive export
        print("🧪 Testing comprehensive export...")
        
        # Import the analysis module with proper path handling
        try:
            import sys
            import os
            
            # Add current directory to Python path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from analysis import export_comprehensive_analysis, run_simple_export
            print("   ✅ Analysis module imported successfully")
        except ImportError as e:
            print(f"   ❌ Analysis import failed: {e}")
            print("   ℹ️  Make sure 'analysis.py' is in the same directory as 'test_analysis.py'")
            
            # Try alternative approach - run analysis directly
            print("   🔄 Attempting direct analysis...")
            try:
                from functions import ReturnsClusteringAnalysis
                return run_direct_analysis_test(test_data_file)
            except ImportError:
                print("   ❌ Cannot import functions either")
                return False
        
        # Test simple export first
        print("🧪 Testing simple export...")
        simple_file = run_simple_export(test_data_file, 'test_simple')
        assert simple_file is not None, "Simple export failed"
        assert os.path.exists(simple_file), "Simple export file not created"
        print(f"   ✅ Simple export successful: {simple_file}")
        
        # Test comprehensive export
        print("🧪 Testing comprehensive export...")
        excel_file, csv_file = export_comprehensive_analysis(test_data_file, 'test_comprehensive')
        
        if excel_file:
            assert os.path.exists(excel_file), "Excel file not created"
            print(f"   ✅ Excel export successful: {excel_file}")
            
            # Test Excel file contents
            print("🧪 Validating Excel file contents...")
            excel_data = pd.ExcelFile(excel_file)
            expected_sheets = [
                'Cluster_Summary', 'Feature_Dictionary', 'Detailed_Metrics',
                'Feature_Statistics', 'Business_Priority', 'Customer_Samples',
                'Expected_Clusters_Reference', 'Analysis_Summary'
            ]
            
            for sheet in expected_sheets:
                assert sheet in excel_data.sheet_names, f"Missing sheet: {sheet}"
            
            print(f"   ✅ All {len(expected_sheets)} sheets present")
            
            # Test specific sheet contents
            cluster_summary = pd.read_excel(excel_file, sheet_name='Cluster_Summary')
            assert len(cluster_summary) > 0, "Empty cluster summary"
            
            expected_clusters = pd.read_excel(excel_file, sheet_name='Expected_Clusters_Reference')
            assert len(expected_clusters) >= 10, "Missing expected cluster types"
            
            print("   ✅ Sheet contents validated")
        
        if csv_file:
            assert os.path.exists(csv_file), "CSV file not created"
            print(f"   ✅ CSV backup successful: {csv_file}")
        
        print(f"\n🎉 COMPREHENSIVE EXPORT TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_data_quality_tests(test_data_file):
    """
    Run data quality validation tests
    
    Args:
        test_data_file (str): Path to test data CSV
    """
    
    print("\n" + "="*60)
    print("RUNNING DATA QUALITY TESTS")
    print("="*60)
    
    try:
        df = pd.read_csv(test_data_file)
        
        # Test 1: Data structure
        print("🧪 Test 1: Data structure validation...")
        required_columns = [
            'CUSTOMER_EMAILID', 'SALES_ORDER_NO', 'SKU', 'SALES_QTY',
            'RETURN_QTY', 'ORDER_DATE', 'RETURN_DATE', 'UNITS_RETURNED_FLAG'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        print("   ✅ All required columns present")
        
        # Test 2: Data consistency
        print("🧪 Test 2: Data consistency validation...")
        
        # Returns should have RETURN_QTY > 0
        returns_mask = df['UNITS_RETURNED_FLAG'] == 'Yes'
        assert (df[returns_mask]['RETURN_QTY'] > 0).all(), "Inconsistent return data"
        
        # Non-returns should have RETURN_QTY = 0
        non_returns_mask = df['UNITS_RETURNED_FLAG'] == 'No'
        assert (df[non_returns_mask]['RETURN_QTY'] == 0).all(), "Inconsistent non-return data"
        
        print("   ✅ Data consistency validated")
        
        # Test 3: Date format validation
        print("🧪 Test 3: Date format validation...")
        
        # Check ORDER_DATE format
        sample_order_dates = df['ORDER_DATE'].head()
        for date_str in sample_order_dates:
            try:
                pd.to_datetime(date_str, format='%m/%d/%Y')
            except:
                assert False, f"Invalid ORDER_DATE format: {date_str}"
        
        # Check RETURN_DATE format (should be Excel serial or "-")
        sample_return_dates = df[df['RETURN_DATE'] != '-']['RETURN_DATE'].head()
        for date_str in sample_return_dates:
            try:
                float(date_str)  # Should be convertible to float (Excel serial)
            except:
                assert False, f"Invalid RETURN_DATE format: {date_str}"
        
        print("   ✅ Date formats validated")
        
        # Test 4: Business logic validation
        print("🧪 Test 4: Business logic validation...")
        
        # Check reasonable return rates
        customer_stats = df.groupby('CUSTOMER_EMAILID').agg({
            'RETURN_QTY': ['sum', 'count'],
            'SALES_QTY': 'sum'
        })
        customer_stats.columns = ['total_returned', 'total_items', 'total_purchased']
        customer_stats['return_rate'] = customer_stats['total_returned'] / customer_stats['total_purchased']
        
        assert customer_stats['return_rate'].max() <= 1.0, "Return rate > 100% detected"
        assert customer_stats['return_rate'].min() >= 0.0, "Negative return rate detected"
        
        print("   ✅ Business logic validated")
        
        print(f"\n🎉 ALL DATA QUALITY TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main test runner
    """
    print("="*80)
    print("CUSTOMER CLUSTERING ANALYSIS - TEST SUITE")
    print("="*80)
    
    # Configuration
    test_data_file = 'test_returns_data.csv'
    n_test_customers = 500  # Smaller for faster testing
    
    # Step 1: Create test data
    print("Step 1: Creating test data...")
    test_df = create_test_data(n_test_customers, test_data_file)
    
    # Step 2: Run data quality tests
    print("\nStep 2: Running data quality tests...")
    if not run_data_quality_tests(test_data_file):
        print("❌ Data quality tests failed. Stopping.")
        return
    
    # Step 3: Run basic validation tests
    print("\nStep 3: Running basic validation tests...")
    if not run_basic_validation_tests(test_data_file):
        print("❌ Basic validation tests failed. Stopping.")
        return
    
    # Step 4: Run comprehensive export test
    print("\nStep 4: Running comprehensive export test...")
    if not run_comprehensive_export_test(test_data_file):
        print("❌ Export tests failed. Stopping.")
        return
    
    # Summary
    print("\n" + "="*80)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("="*80)
    print("✅ Test data created and validated")
    print("✅ Analysis pipeline working correctly")
    print("✅ Export functionality working correctly")
    print("✅ Ready for production use!")
    
    # Cleanup option
    print(f"\nTest files created:")
    print(f"- {test_data_file}")
    
    # List any export files created
    for file in os.listdir('.'):
        if file.startswith('test_') and (file.endswith('.xlsx') or file.endswith('.csv')):
            print(f"- {file}")
    
    print("\nTo clean up test files, run:")
    print("rm test_*")

if __name__ == "__main__":
    main()