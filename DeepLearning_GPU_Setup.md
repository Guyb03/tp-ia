To run Deep Learning TPs, which requires specific resources and packages, follow these instructions based on whether you're using a lab machine or your local machine.

For Lab Machine Users:

1. Open Visual Studio Code and locate the Remote-SSH Extension.
2. In the left panel of VSCode, find the Remote icon and add an SSH connection (alternatively, you can click on the green flash icon at the bottom left of the VSCode window).
	1. Use the following SSH connection command: ssh <your_ensimag_or_uga_login>@vmgpu<x.x.x>.ensigmag.fr (replace x.x.x with a number between 001 and 064).
	2. Enter your password when prompted.
3. If the connection is successful, you'll see "SSH:vmgpuxxx…" in the green box at the bottom left of the VSCode window.
4. If this is your first time using VSCode on the lab machine, you'll need to install ipykernel.
5. You're now ready to work on Deep Learning with GPU support.

For Local Machine Users:

1. Use the Ensimag VPN by visiting this link: <https://intranet.ensimag.grenoble-inp.fr/fr/informatique/vpn-ensimag>
2. Activate the VPN and connect to the virtual machine via VSCode, following the instructions provided for lab machine users.
