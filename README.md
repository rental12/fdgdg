# GPU pass through Docker WSL2 

## Installing Developer Windows

In the start bar, search for "Windows Insider program settings". In this setting, you can choose the insider build to install. To use GPU with WSL2, you need to enroll your device to the Dev channel. Click on the build and select dev channel. Finally, update your windows and your device will be running the latest developer build.
You will also need enable "receive updates for other microsoft products" in the advance settings to get the up to WSL 2 version.

## Install WSL 2

Installing WSL 2 is simple, open up your powershell as administrator and type the command to install WSL 1 first
```bash 
wsl --install
```
and to enable WSL 1, type 
```bash
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```
and finally, to enable wsl 2, type 
```bash
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```
restart your computer after this step. To ensure you are running wsl 2, type
```bash
wsl.exe --set-default-version 2
```

## Install and setting up Ubuntu on Windows
To install Ubuntu windows, head over to the Microsoft store and search for Ubuntu (the one without any version eg 1804) and click install. After installation is complete, open up Ubuntu and set your sudo password. We will stop here for now to install the Nvidia drivers necessary for later part.

## Install Nvidia Drivers
Before continuing, Nvidia driver support for WSL2 has to be installed in your host pc (windows). To do so , visit https://developer.nvidia.com/cuda/wsl/download and click on the appropriate drivers to download depending on what type of gpu you have. To check if the installation is successful, you can download the nvidia driver inside WSL2 ubuntu by installing the nvidia-utils. Check the version to install by typing nvidia-smi and since nvidia-smi is not installed, you will be prompted to install nvidia utils. Check for the appropriate version in my case i download via
```bash
sudo apt install nvidia-utils-435
```

## Install docker in wsl2

 

