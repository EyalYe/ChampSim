# temp infer just to make sure everything is working

import argparse

def main():
    parser = argparse.ArgumentParser(description="Infer script to check setup.")
    parser.add_argument('--test', action='store_true', help='Run a test inference.')
    parser.add_argument('--input_file', type=str, help='Path to the input file for inference.')
    parser.add_argument('--output_file', type=str, help='Path to save the inference output.')
    args = parser.parse_args()

    input_file = args.input_file if args.input_file else "default_input.csv"
    output_file = args.output_file if args.output_file else "default_output.txt"

    
    with open(input_file, 'r') as f:
        data = f.read()
        with open(output_file, 'w') as out:
            try:
                hex_data = data.splitlines()[-1].split(',')[0]
            except IndexError:
                print("Error parsing the last line of the input file.")
                hex_data = '0x0'
                print(data.splitlines())
            except Exception as e:
                print(f"Unexpected error: {e}")
                hex_data = '0x0'
                print(data)
            try:
                number = int(hex_data, 16) + 15
            except Exception as e:
                print(f"Error converting hex to int: {e}")
                print(data)
                number = 0
            out.write(f"0x{number:x}")

if __name__ == "__main__":
    main()
    

