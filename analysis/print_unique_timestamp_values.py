import os
import glob

def print_unique_values():
    """
    Extract unique values from Type, Channel, and Label columns in .txt files.
    """
    
    # Set to store unique values
    unique_types = set()
    unique_channels = set()
    unique_labels = set()
    
    # Get all .txt files in the data_raw folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.abspath(os.path.join(script_dir, "..", "data_raw"))
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    print(f"\nFound {len(txt_files)} .txt files to analyze...\n")
    print("=" * 50)
    
    # Process each file
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
                # Skip header lines (first 2 lines)
                data_lines = lines[2:]
                
                for line in data_lines:
                    line = line.strip()
                    if line:  # Skip empty lines
                        # Split by tab
                        columns = line.split('\t')
                        
                        if len(columns) >= 5:  # Ensure we have enough columns
                            # Extract values (Index, Time, Type, Channel, Label, Date Created)
                            type_val = columns[2].strip()
                            channel_val = columns[3].strip()
                            label_val = columns[4].strip()
                            
                            # Add to sets
                            unique_types.add(type_val)
                            unique_channels.add(channel_val)
                            unique_labels.add(label_val)
                            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Print results
    print("UNIQUE VALUES FOUND:")
    print("=" * 50)
    
    print(f"\nType values ({len(unique_types)} unique):")
    for type_val in sorted(unique_types):
        print(f"  - {type_val}")
    
    print(f"\nChannel values ({len(unique_channels)} unique):")
    for channel_val in sorted(unique_channels):
        print(f"  - {channel_val}")
    
    print(f"\nLabel values ({len(unique_labels)} unique):")
    for label_val in sorted(unique_labels):
        print(f"  - {label_val}")
    
    print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    print_unique_values()
