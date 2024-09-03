import os
import paramiko
from scp import SCPClient

def create_ssh_client(host, port, username, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port, username, password)
    return client

def copy_and_rename_excel_files(ssh_client, remote_source_dir, local_dest_dir):
    # Create the destination directory if it does not exist
    os.makedirs(local_dest_dir, exist_ok=True)

    # Create an SCP client
    with SCPClient(ssh_client.get_transport()) as scp:
        # Use the SSH client to find all .xlsx files in the remote source directory
        stdin, stdout, stderr = ssh_client.exec_command(f"find {remote_source_dir} -type f -name '*.xlsx'")
        #stdin, stdout, stderr = ssh_client.exec_command(f"find {remote_source_dir} -type f -name '*.txt'")
        files = stdout.read().decode().splitlines()

        for file in files:
            # Extract the relevant parts of the file path
            parts = file.split('/')
            if True and len(parts) >= 4:
                new_file_name = parts[-1] #parts[-3] + '_' + parts[-1]
                new_file_name = parts[-3] + '_' + parts[-1]
                local_file_path = os.path.join(local_dest_dir, new_file_name)

                if not os.path.exists(local_file_path):
                    # Copy the file to the local destination directory with the new name
                    scp.get(file, local_file_path)
                    print(f'Copied {file} to {local_file_path}')
                else:
                    print(f"{new_file_name} exists")

def main():
    # Define SSH connection details for the data transfer node and the remote host
    TRANSFER_NODE_HOST = 'cc-xfer.campuscluster.illinois.edu'
    TRANSFER_NODE_PORT = 22

    REMOTE_HOST = 'cc-login.campuscluster.illinois.edu'
    REMOTE_PORT = 22
    REMOTE_USER = 'arjitm2'
    REMOTE_PASS = 'BandarVanmanush8!'
    REMOTE_SOURCE_DIR = '/projects/illinois/ovcri/beckman/aandrsn3/eeg_data/*/' #*/'
    LOCAL_DEST_DIR = 'qeeg_icare/'

    # Create an SSH client for the data transfer node
    transfer_node_client = create_ssh_client(TRANSFER_NODE_HOST, TRANSFER_NODE_PORT, REMOTE_USER, REMOTE_PASS)

    # Set up SSH tunneling from the data transfer node to the remote host
    transport = transfer_node_client.get_transport()
    dest_addr = (REMOTE_HOST, REMOTE_PORT)
    local_addr = ('localhost', 22)
    tunnel = transport.open_channel("direct-tcpip", dest_addr, local_addr)

    # Create an SSH client for the remote host using the tunnel
    remote_client = paramiko.SSHClient()
    remote_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    remote_client.connect('localhost', port=22, username=REMOTE_USER, password=REMOTE_PASS, sock=tunnel)

    # Copy and rename the Excel files
    copy_and_rename_excel_files(remote_client, REMOTE_SOURCE_DIR, LOCAL_DEST_DIR)

    # Close the SSH connections
    remote_client.close()
    transfer_node_client.close()

if __name__ == "__main__":
    main()


