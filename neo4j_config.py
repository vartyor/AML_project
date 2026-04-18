from __future__ import annotations

# Neo4j 연결 설정

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "neo4j"   # ← 실제 비밀번호로 변경

# LangChain Neo4j용 URL (bolt → neo4j+s 프로토콜 자동 처리)
NEO4J_URL      = NEO4J_URI


# 연결 테스트 유틸

def get_driver():
    """
    neo4j.GraphDatabase.driver 인스턴스를 반환합니다.
    연결 실패 시 ConnectionError를 발생시킵니다.
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def test_connection() -> bool:
    """
    Neo4j 연결을 테스트하고 결과를 출력합니다.

    Returns:
        bool: 연결 성공 여부
    """
    print(f"[Neo4j] 연결 시도: {NEO4J_URI}")
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run("RETURN 'AML GraphRAG 연결 성공 ✅' AS msg")
            msg = result.single()["msg"]
            print(f"[Neo4j] {msg}")

        # 현재 노드/관계 수 확인
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            rel_count  = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
            print(f"[Neo4j] 현재 DB 상태 — 노드: {node_count}개 / 관계: {rel_count}개")

        driver.close()
        return True

    except Exception as e:
        print(f"[Neo4j] ❌ 연결 실패: {e}")
        print("\n확인 사항:")
        print("  1. Neo4j Desktop에서 인스턴스가 'Started' 상태인지 확인")
        print("  2. neo4j_config.py의 NEO4J_PASSWORD가 올바른지 확인")
        print("  3. bolt://localhost:7687 포트가 열려 있는지 확인")
        return False

# 직접 실행 시 연결 테스트

if __name__ == "__main__":
    test_connection()
