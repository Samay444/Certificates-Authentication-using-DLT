import hashlib
import time
import json
import tkinter as tk
from tkinter import filedialog
from typing import List, Optional
import dataclasses
from dataclasses import dataclass, asdict
import difflib  # For text comparison
from PIL import Image
import imagehash  # To compare images via perceptual hash

@dataclass
class Block:
    index: int
    timestamp: float
    certificate_hash: str
    previous_hash: str
    nonce: int = 0  # Used for mining (Proof of Work)

    def compute_hash(self) -> str:
        """Generate a SHA-256 hash for the block based on its contents."""
        block_string = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    difficulty: int = 2  # Adjust difficulty of mining (higher number means harder)

    def __init__(self):
        """Initialize the blockchain with the genesis block."""
        self.chain: List[Block] = []
        self.create_genesis_block()

    def create_genesis_block(self):
        """Create the first block of the blockchain (genesis block)."""
        genesis_block = Block(0, time.time(), "0", "0")
        genesis_block.hash = self.proof_of_work(genesis_block)
        self.chain.append(genesis_block)

    def proof_of_work(self, block: Block) -> str:
        """Simple Proof-of-Work to add computational difficulty to the block creation."""
        block.nonce = 0
        computed_hash = block.compute_hash()
        while not computed_hash.startswith("0" * Blockchain.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()
        return computed_hash

    def add_block(self, certificate_hash: str) -> Block:
        """Add a new block to the chain with a hash of the certificate data."""
        previous_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            certificate_hash=certificate_hash,
            previous_hash=previous_block.compute_hash(),
        )
        new_block.hash = self.proof_of_work(new_block)
        self.chain.append(new_block)
        return new_block

    def is_chain_valid(self) -> bool:
        """Verify the integrity of the blockchain."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if current.hash != current.compute_hash():
                print(f"Block {i} has been altered")
                return False
            if current.previous_hash != previous.hash:
                print(f"Block {i} is disconnected from Block {i-1}")
                return False
        return True

    def find_certificate(self, certificate_hash: str) -> Optional[Block]:
        """Search for a certificate hash in the blockchain."""
        for block in self.chain:
            if block.certificate_hash == certificate_hash:
                return block
        return None

# Helper function to simulate hashing a certificate (HTML content here)
def generate_certificate_hash(file_content: str) -> str:
    """Simulate creating a hash for a certificate using SHA-256."""
    return hashlib.sha256(file_content.encode('utf-8')).hexdigest()

# Helper function to calculate similarity for text (HTML/MHTML files)
def compute_similarity(file1_content: str, file2_content: str) -> float:
    """Compare two text contents and compute the similarity percentage."""
    sequence_matcher = difflib.SequenceMatcher(None, file1_content, file2_content)
    similarity = sequence_matcher.ratio()  # Returns a float between 0 and 1
    return similarity * 100  # Convert to percentage

# Helper function to calculate image hash and compare similarity
def compare_images(image1_path: str, image2_path: str) -> float:
    """Compute similarity percentage between two images using perceptual hashing."""
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Use imagehash to generate perceptual hash
    hash1 = imagehash.phash(image1)
    hash2 = imagehash.phash(image2)

    # Calculate the Hamming distance (difference between hashes)
    hash_diff = hash1 - hash2
    max_diff = len(hash1.hash) * hash1.hash.shape[0]  # Maximum possible difference (binary length)
    similarity = 100 * (1 - hash_diff / max_diff)  # Convert to percentage similarity

    return similarity

def load_html_files() -> Optional[List[str]]:
    """Load two HTML/MHTML files one by one via file dialog and return their contents."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Select first file
    print("Please select the first HTML or MHTML file.")
    file_path1 = filedialog.askopenfilename(
        title="Select the first HTML or MHTML file",
        filetypes=[("HTML files", "*.html;*.mhtml")]
    )
    
    if not file_path1:  # If no file is selected, return None
        print("No file selected.")
        return None
    
    # Select second file
    print("Please select the second HTML or MHTML file.")
    file_path2 = filedialog.askopenfilename(
        title="Select the second HTML or MHTML file",
        filetypes=[("HTML files", "*.html;*.mhtml")]
    )
    
    if not file_path2:  # If no second file is selected, return None
        print("No second file selected.")
        return None
    
    print(f"Selected file paths: {file_path1}, {file_path2}")
    
    # Read both files' contents
    file_contents = []
    for file_path in [file_path1, file_path2]:
        with open(file_path, "r", encoding="utf-8") as file:
            file_contents.append(file.read())  # Read file content

    return file_contents

def load_image_files() -> Optional[List[str]]:
    """Load two image files one by one via file dialog and return their paths."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Select first image
    print("Please select the first image file.")
    image_path1 = filedialog.askopenfilename(
        title="Select the first image file",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    
    if not image_path1:  # If no image is selected, return None
        print("No image selected.")
        return None
    
    # Select second image
    print("Please select the second image file.")
    image_path2 = filedialog.askopenfilename(
        title="Select the second image file",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    
    if not image_path2:  # If no second image is selected, return None
        print("No second image selected.")
        return None
    
    print(f"Selected image paths: {image_path1}, {image_path2}")
    return [image_path1, image_path2]

# Example usage
if __name__ == "__main__":
    # Initialize blockchain
    blockchain = Blockchain()
    
    # Load two HTML/MHTML files using file dialog
    html_files_data = load_html_files()
    if html_files_data:
        file1_content = html_files_data[0]
        file2_content = html_files_data[1]

        # Generate and add the first file's certificate hash to the blockchain
        cert_hash_1 = generate_certificate_hash(file1_content)
        blockchain.add_block(cert_hash_1)
        print(f"Certificate hash for first HTML/MHTML file added to blockchain: {cert_hash_1}")

        # Generate and add the second file's certificate hash to the blockchain
        cert_hash_2 = generate_certificate_hash(file2_content)
        blockchain.add_block(cert_hash_2)
        print(f"Certificate hash for second HTML/MHTML file added to blockchain: {cert_hash_2}")

        # Verify blockchain integrity
        print("\nBlockchain validity check:", blockchain.is_chain_valid())

        # Calculate similarity percentage for HTML/MHTML files
        similarity_percentage_html = compute_similarity(file1_content, file2_content)
        print(f"Authenticity: {similarity_percentage_html:.2f}%")
        if(similarity_percentage_html > 90):
            print("Original")
        elif(90 > similarity_percentage_html >50):
            print("Tampered")
        else:
            print("Forged")


    # Load two image files using file dialog
    image_files_data = load_image_files()
    if image_files_data:
        image1_path = image_files_data[0]
        image2_path = image_files_data[1]

        # Calculate similarity percentage for image files
        similarity_percentage_img = compare_images(image1_path, image2_path)
        print(f"Authenticity: {similarity_percentage_img:.2f}%")
        if(similarity_percentage_img > 90):
            print("Original")
        elif(90 > similarity_percentage_img >50):
            print("Tampered")
        else:
            print("Forged")
            