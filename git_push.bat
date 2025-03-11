:: Add tất cả các file
echo === Adding files ===
git add .

:: Nhập commit message
set /p message="Update"
echo === Committing with message: %message% ===
git commit -m "%message%"

:: Push lên remote repository
echo === Pushing to remote ===
git push

echo === Done ===