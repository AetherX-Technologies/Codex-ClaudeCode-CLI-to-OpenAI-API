import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request


def post_json(url: str, payload: dict, token: str, timeout: float) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body
    except Exception as e:  # pragma: no cover
        return -1, str(e)


def run_batch(count: int, endpoint: str, model: str, token: str, base_url: str, timeout: float) -> None:
    url = f"{base_url}/{endpoint}"
    print(f"Hitting {url} with {count} concurrent requests")
    payload = (
        {"model": model, "messages": [{"role": "user", "content": "Hello"}]}
        if endpoint == "v1/chat/completions"
        else {"model": model, "prompt": "Hello"}
    )
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futs = [pool.submit(post_json, url, payload, token, timeout) for _ in range(count)]
        for idx, fut in enumerate(concurrent.futures.as_completed(futs), start=1):
            status, body = fut.result()
            preview = (body or "").replace("\n", " ")[:120]
            print(f"[{idx}/{count}] status={status} body={preview}")
    print(f"Elapsed: {time.time() - started:.2f}s for {count} requests")


def main():
    default_base = os.getenv(
        "API_BASE_URL",
        f"http://{os.getenv('API_HOST', '127.0.0.1')}:{os.getenv('API_PORT', '8000')}",
    )
    parser = argparse.ArgumentParser(description="Fire concurrent requests to Codex API.")
    parser.add_argument("--count", type=int, default=6, help="Number of concurrent requests to send")
    parser.add_argument("--endpoint", choices=["v1/chat/completions", "v1/completions"], default="v1/chat/completions")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--base-url", default=default_base)
    parser.add_argument("--token", default=os.getenv("API_AUTH_TOKEN", "local-token"))
    args = parser.parse_args()

    run_batch(
        count=args.count,
        endpoint=args.endpoint,
        model=args.model,
        token=args.token,
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
