use python 3.10

open powershell as administrator 
Get-ExecutionPolicy   -> enter
if policy restricted
Set-ExecutionPolicy RemoteSigned  -> enter
select Yes 'Y' -> enter

for create python environment
python -m venv venv

if Command Prompt
venv\Scripts\activate

if Powershell
venv\Scripts\Activate.ps1

if python virtual environment activate successfully
(venv) PS C:\xampp\htdocs\myproject> what and where you created

pip install binance-connector mysql-connector-python python-dotenv pandas pandas-ta numpy

if numpy version error
pip uninstall numpy
pip install numpy==1.23.5
pip install --upgrade pandas-ta



