# scripts/test_retrieval_service.py
import argparse

from verl.tools.utils.search_r1_like_utils import call_search_api, perform_single_search_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--query", default="What is the capital of France?")
    args = ap.parse_args()

    # 1) 轻量：只看 HTTP/JSON 是否通
    resp, err = call_search_api(args.url, [args.query], topk=args.topk, timeout=args.timeout)
    print("call_search_api err:", err)
    if resp is not None:
        print("call_search_api resp keys:", list(resp.keys()))

    # 2) 和训练一致：看格式化后的 result_text
    result_text, meta = perform_single_search_batch(args.url, args.query, topk=args.topk, timeout=args.timeout)
    print("perform_single_search_batch meta:", {k: meta.get(k) for k in ["status", "total_results", "api_request_error"]})
    print("result_text:", result_text[:2000])


if __name__ == "__main__":
    main()