# download_data.ps1
$url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
$output = "dataset.txt"

Write-Host "Downloading TinyStories (This may take a minute)..."
Invoke-WebRequest -Uri $url -OutFile $output
Write-Host "Download Complete: $output"

# Create a small validation split (first 1000 lines)
Get-Content $output -TotalCount 1000 | Set-Content "dataset_val.txt"
