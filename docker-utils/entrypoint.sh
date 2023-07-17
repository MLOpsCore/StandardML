# Parse types argument as --types {json}
# and write to types.json
echo "Parsing config argument: $2"
echo "$2" | jq . >config.json

python -m standardml --config config.json
