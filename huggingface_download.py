import subprocess

def install_huggingface_cli():
  """
  Installs the Hugging Face CLI if not already installed.
  """
  try:
    subprocess.run(["pip", "install", "huggingface_hub[cli]"])
    print("Hugging Face CLI installed successfully!")
  except subprocess.CalledProcessError:
    print("Error installing Hugging Face CLI. Please install manually using pip install huggingface_hub[cli]")

def download_model(model_id, include_pattern, local_dir):
  """
  Downloads the original weights for a model using the Hugging Face CLI.

  Args:
    model_id: The ID of the model on Hugging Face Hub (e.g., "meta-llama/Meta-Llama-3-8B-Instruct").
    include_pattern: A globbing pattern to specify files to download (e.g., "original/*").
    local_dir: The local directory to save the downloaded files.
  """
  command = f"huggingface-cli download {model_id} --include {include_pattern} --local-dir {local_dir}"
  #command = f"huggingface-cli download {model_id} --local-dir {local_dir}"
  try:
    subprocess.run(command.split(), check=True)
    print(f"Model weights downloaded to {local_dir}")
  except subprocess.CalledProcessError:
    print(f"Error downloading model. Please check the model ID and network connectivity.")

if __name__ == "__main__":
  # Replace with your desired model ID
  model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
  #model_id = "LoneStriker/Meta-Llama-3-8B-Instruct-8.0bpw-h8-exl2"
  include_pattern = "original/*"
  local_dir = "Meta-Llama-3-8B-Instruct"
  try:
      subprocess.run(["huggingface-cli", "--version"], check=True, capture_output=True, text=True)
  except subprocess.SubprocessError as e:
      print("Hugging Face CLI not found or another error occurred. Trying to install...")
      install_huggingface_cli()

  download_model(model_id, include_pattern, local_dir)
