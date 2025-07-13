import subprocess
import datetime
import os

def run(cmd):
    print(f">>> 실행: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠️ 에러 발생: {cmd}")
    return result.returncode

def git_main():
    #os.chdir('../../')

    # 0. 워킹 디렉토리가 깨끗한지 확인
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    is_clean = result.stdout.strip() == ""

    if not is_clean:
        print("🔄 변경 사항이 있으므로 stash")
        run("git stash -u")

    # 1. pull
    if run("git pull origin main") != 0:
        print("❌ pull 실패. rebase 중이거나 충돌 가능성 있음")
        return

    if not is_clean:
        print("🧺 stash pop")
        run("git stash pop")

    # 2. add
    run("git add .")

    # 3. 커밋할 내용이 있는지 확인
    result = subprocess.run("git diff --cached --quiet", shell=True)
    if result.returncode != 0:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"Auto commit at {now}"
        run(f'git commit -m "{commit_msg}"')
        run("git push origin main")
    else:
        print("✅ 커밋할 변경사항 없음 (working tree clean)")


def git_pull():
    #os.chdir('../../')
    print(">>> GitHub에서 최신 내용 가져오는 중...")

    cmd = "git pull origin main"  # 브랜치 이름은 필요 시 수정

    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        print("✅ 최신 내용으로 갱신되었습니다.")
    else:
        print("⚠️ git pull 중 에러가 발생했습니다.")

#git_main()