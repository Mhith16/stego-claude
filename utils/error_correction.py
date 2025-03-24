import numpy as np
import torch
import re
import hashlib
import zlib

class ReedSolomonEncoder:
    """
    Implementation of Reed-Solomon error correction for binary data
    This is a simplified implementation for demonstration purposes
    In production, you would use a library like reedsolo
    """
    def __init__(self, n=255, k=223):
        """
        Initialize with parameters:
        n: codeword length
        k: message length
        """
        self.n = n
        self.k = k
        self.redundancy = n - k
        
    def encode(self, data):
        """
        Encode binary data with Reed-Solomon
        For a complete implementation, use a proper RS library
        """
        # In a real implementation, this would use Galois Field arithmetic
        # For demo purposes, we'll use a simple parity-based approach
        if len(data) > self.k:
            # Truncate if too long
            data = data[:self.k]
        elif len(data) < self.k:
            # Pad with zeros if too short
            data = np.pad(data, (0, self.k - len(data)))
            
        # Create simple parity bits (in real RS, this would be more complex)
        parity = []
        for i in range(self.redundancy):
            # XOR subsets of the data as a simple form of parity
            indices = [j for j in range(self.k) if (j & (i+1)) != 0]
            parity_bit = 0
            for idx in indices:
                parity_bit ^= data[idx]
            parity.append(parity_bit)
            
        # Combine data and parity
        encoded = np.concatenate([data, np.array(parity)])
        return encoded
        
    def decode(self, received):
        """
        Decode Reed-Solomon encoded data
        For a complete implementation, use a proper RS library
        """
        # Extract data and parity parts
        data_part = received[:self.k]
        parity_part = received[self.k:]
        
        # Check parity and correct errors if possible
        corrected_data = data_part.copy()
        
        # Calculate syndrome (in real RS, this would use proper GF arithmetic)
        syndrome = []
        for i in range(self.redundancy):
            indices = [j for j in range(self.k) if (j & (i+1)) != 0]
            parity_check = 0
            for idx in indices:
                parity_check ^= data_part[idx]
            syndrome.append(parity_check ^ parity_part[i])
        
        # If syndrome is all zeros, no errors detected
        if sum(syndrome) == 0:
            return corrected_data, 1.0  # High confidence
            
        # Simple error correction for demonstration
        # In a real implementation, this would use Berlekamp-Massey algorithm
        try:
            # Try to correct single-bit errors
            if sum(syndrome) == 1:
                error_pos = syndrome.index(1)
                if error_pos < self.k:
                    corrected_data[error_pos] ^= 1  # Flip the erroneous bit
                confidence = 0.9  # High confidence for single error
            else:
                # For multi-bit errors, use heuristic
                confidence = max(0.5, 1.0 - (sum(syndrome) / self.redundancy))
        except:
            confidence = 0.5  # Medium confidence
            
        return corrected_data, confidence


class RedundantPatientDataProcessor:
    """
    Process and encode/decode patient data with multiple redundancy layers
    """
    def __init__(self, max_message_length=1024, redundancy_copies=3):
        self.max_message_length = max_message_length
        self.rs_encoder = ReedSolomonEncoder(n=127, k=99)  # Smaller chunks for better error correction
        self.redundancy_copies = redundancy_copies  # Number of copies for critical data
        
    def compute_crc(self, data_string):
        """Compute CRC32 checksum for a string"""
        return zlib.crc32(data_string.encode())
    
    def encode_field_with_redundancy(self, field_name, field_value):
        """Encode a single field with multiple redundancy layers"""
        # 1. Convert to ASCII binary
        field_binary = ''.join([format(ord(c), '08b') for c in field_value])
        field_bits = np.array([int(bit) for bit in field_binary], dtype=np.float32)
        
        # 2. Calculate checksum
        checksum = self.compute_crc(field_value)
        checksum_binary = format(checksum, '032b')  # 32-bit CRC
        checksum_bits = np.array([int(bit) for bit in checksum_binary], dtype=np.float32)
        
        # 3. Create field header (field name encoded)
        header_binary = ''.join([format(ord(c), '08b') for c in field_name])
        header_bits = np.array([int(bit) for bit in header_binary], dtype=np.float32)
        
        # 4. Combine field data, header, and checksum
        field_chunk = np.concatenate([header_bits, field_bits, checksum_bits])
        
        # 5. Apply Reed-Solomon encoding
        encoded_chunk = self.rs_encoder.encode(field_chunk)
        
        return encoded_chunk
        
    def encode_patient_data(self, patient_data):
        """
        Encode patient data text into binary format with multiple redundancy layers
        Format expected:
        Name: [NAME]
        Age: [AGE]
        ID: [ID]
        """
        # Extract fields using regex
        name_match = re.search(r'Name:\s*(.*?)(?:\n|$)', patient_data)
        age_match = re.search(r'Age:\s*(.*?)(?:\n|$)', patient_data)
        id_match = re.search(r'ID:\s*(.*?)(?:\n|$)', patient_data)
        
        # Get values or defaults
        name = name_match.group(1).strip() if name_match else ""
        age = age_match.group(1).strip() if age_match else ""
        patient_id = id_match.group(1).strip() if id_match else ""
        
        # Primary encoding for each field
        name_encoded = self.encode_field_with_redundancy("NAME", name)
        age_encoded = self.encode_field_with_redundancy("AGE", age)
        id_encoded = self.encode_field_with_redundancy("ID", patient_id)
        
        # Create redundant copies of critical fields (name and ID)
        all_chunks = []
        
        # For critical fields (name and ID), create multiple copies
        for _ in range(self.redundancy_copies):
            all_chunks.append(name_encoded)
            all_chunks.append(id_encoded)
        
        # Age is less critical - add fewer copies
        all_chunks.append(age_encoded)
        if self.redundancy_copies > 1:
            all_chunks.append(age_encoded)  # Just one extra copy
            
        # Also add a complete record in another format for redundancy
        # Full record with all fields together
        full_record = f"NAME:{name};AGE:{age};ID:{patient_id}"
        full_binary = ''.join([format(ord(c), '08b') for c in full_record])
        full_bits = np.array([int(bit) for bit in full_binary], dtype=np.float32)
        
        # CRC for the full record
        full_checksum = self.compute_crc(full_record)
        full_checksum_binary = format(full_checksum, '032b')
        full_checksum_bits = np.array([int(bit) for bit in full_checksum_binary], dtype=np.float32)
        
        # Combine and encode with RS
        full_chunk = np.concatenate([full_bits, full_checksum_bits])
        encoded_full = self.rs_encoder.encode(full_chunk)
        
        # Add to chunks
        all_chunks.append(encoded_full)
            
        # Flatten all chunks
        all_bits = np.concatenate(all_chunks)
        
        # Ensure fixed length by padding or truncating
        if len(all_bits) > self.max_message_length:
            print(f"Warning: Encoded data length ({len(all_bits)}) exceeds maximum message length ({self.max_message_length}). Truncating.")
            all_bits = all_bits[:self.max_message_length]
        else:
            all_bits = np.pad(all_bits, (0, self.max_message_length - len(all_bits)))
            
        return torch.tensor(all_bits, dtype=torch.float32)
    
    def decode_patient_data(self, binary_tensor):
        """
        Decode binary tensor back to patient data format with redundancy handling
        Returns the decoded text and confidence scores for each field
        """
        # Convert to numpy for easier processing
        if torch.is_tensor(binary_tensor):
            binary_array = binary_tensor.detach().cpu().numpy()
        else:
            binary_array = binary_tensor
            
        # Convert soft values to hard decisions
        binary_array = (binary_array > 0.5).astype(np.int32)
        
        # Segment the binary array into chunks for processing
        chunk_size = self.rs_encoder.n
        chunks = []
        
        for i in range(0, len(binary_array) - chunk_size, chunk_size):
            chunk = binary_array[i:i+chunk_size]
            if len(chunk) == chunk_size:  # Ensure full chunks
                chunks.append(chunk)
        
        # Decode each chunk
        decoded_data = []
        for chunk in chunks:
            try:
                # Try RS decoding
                data, confidence = self.rs_encoder.decode(chunk)
                decoded_data.append((data, confidence))
            except Exception as e:
                # Skip corrupted chunks
                continue
        
        # Process decoded chunks to find fields
        name_candidates = []
        age_candidates = []
        id_candidates = []
        full_record_candidates = []
        
        for data, confidence in decoded_data:
            try:
                # Try to find header markers
                # Header is at the beginning, field data follows, then checksum
                header_length = 32  # 4 characters * 8 bits
                
                # Extract header, convert to ASCII, and check which field it is
                header_bits = data[:header_length]
                header_bytes = [header_bits[j:j+8] for j in range(0, len(header_bits), 8) if j+8 <= len(header_bits)]
                
                if len(header_bytes) >= 4:  # Ensure we have enough for a header
                    header_chars = ''.join([chr(int(''.join(map(str, byte)), 2)) for byte in header_bytes])
                    
                    # Extract field value
                    field_bits = data[header_length:-32]  # Exclude checksum
                    field_bytes = [field_bits[j:j+8] for j in range(0, len(field_bits), 8) if j+8 <= len(field_bits)]
                    field_text = ''.join([chr(int(''.join(map(str, byte)), 2)) for byte in field_bytes if sum(byte) > 0])
                    
                    # Extract checksum
                    checksum_bits = data[-32:]
                    checksum = int(''.join(map(str, checksum_bits)), 2)
                    
                    # Verify checksum
                    calculated_checksum = self.compute_crc(field_text)
                    checksum_valid = (checksum == calculated_checksum)
                    
                    # Adjust confidence based on checksum
                    if checksum_valid:
                        adjusted_confidence = confidence
                    else:
                        adjusted_confidence = confidence * 0.7  # Reduce confidence if checksum fails
                    
                    # Store candidate based on header
                    if "NAME" in header_chars:
                        name_candidates.append((field_text, adjusted_confidence))
                    elif "AGE" in header_chars:
                        age_candidates.append((field_text, adjusted_confidence))
                    elif "ID" in header_chars:
                        id_candidates.append((field_text, adjusted_confidence))
                    else:
                        # Try to decode as full record
                        if ";" in field_text and ":" in field_text:
                            full_record_candidates.append((field_text, adjusted_confidence))
            except:
                # Skip corrupted chunks
                continue
        
        # Process full record candidates
        for record, confidence in full_record_candidates:
            try:
                # Parse full record
                parts = record.split(';')
                for part in parts:
                    if ":" in part:
                        field_type, value = part.split(':', 1)
                        if "NAME" in field_type and value:
                            name_candidates.append((value, confidence * 0.9))  # Slightly lower confidence for backup format
                        elif "AGE" in field_type and value:
                            age_candidates.append((value, confidence * 0.9))
                        elif "ID" in field_type and value:
                            id_candidates.append((value, confidence * 0.9))
            except:
                continue
        
        # Select best candidate for each field based on confidence and frequency
        def select_best_candidate(candidates):
            if not candidates:
                return "", 0.0
                
            # Count occurrences of each value
            value_counts = {}
            value_confidences = {}
            
            for value, conf in candidates:
                if value not in value_counts:
                    value_counts[value] = 0
                    value_confidences[value] = 0
                
                value_counts[value] += 1
                value_confidences[value] += conf
            
            # Calculate weighted score (frequency * average confidence)
            scores = {}
            for value in value_counts:
                avg_conf = value_confidences[value] / value_counts[value]
                scores[value] = value_counts[value] * avg_conf
            
            # Select value with highest score
            best_value = max(scores.items(), key=lambda x: x[1])[0]
            best_confidence = value_confidences[best_value] / value_counts[best_value]
            
            return best_value, best_confidence
        
        # Get best candidates
        name, name_confidence = select_best_candidate(name_candidates)
        age, age_confidence = select_best_candidate(age_candidates)
        patient_id, id_confidence = select_best_candidate(id_candidates)
        
        # Format the result
        result_text = f"Name: {name}\nAge: {age}\nID: {patient_id}"
        
        # Confidence scores for each field
        confidence = {
            'name': name_confidence,
            'age': age_confidence,
            'id': id_confidence,
            'overall': (name_confidence + age_confidence + id_confidence) / 3
        }
        
        return result_text, confidence

# For backward compatibility
PatientDataProcessor = RedundantPatientDataProcessor