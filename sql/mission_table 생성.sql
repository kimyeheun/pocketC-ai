CREATE DATABASE IF NOT EXISTS pocketC;
use pocketc;

CREATE TABLE IF NOT EXISTS major_categories (
    major_id   INT AUTO_INCREMENT PRIMARY KEY,
    major_name VARCHAR(50) NOT NULL,
    major_type ENUM('고정비','변동비') NOT NULL
);

CREATE TABLE IF NOT EXISTS sub_categories (
    sub_id   INT AUTO_INCREMENT PRIMARY KEY,
    major_id INT NOT NULL,
    sub_name VARCHAR(50) NOT NULL,
    is_target BOOLEAN NOT NULL,
    FOREIGN KEY (major_id) REFERENCES major_categories(major_id)
);

CREATE TABLE IF NOT EXISTS transactions (
	`transaction_id` INT AUTO_INCREMENT PRIMARY KEY, -- 결제내역 아이디
	`user_id`	INT	NULL,             -- 유저 아이디
	`sub_id`	int	NOT NULL,         -- 소분류 카테고리 
	`major_id`	int	NOT NULL,         -- 대분류 카테고리
	`transacted_at`	TIMESTAMP	NULL, -- 거래 시간
	`amount`	INT	NULL,             -- 금액
	`merchanr_name`	VARCHAR(255)	NULL,  -- 거래처
	`staus`	enum('반영', '미반영')	NULL,      -- 반영 / 미반영
	`created_at`	TIMESTAMP	NULL,      -- 생성일
	`updated_at`	TIMESTAMP	NULL       -- 수정일
);

-- (권장) 인덱스: 트랜잭션 조회/집계 성능용
CREATE INDEX idx_tx_user_time   ON transactions (user_id, transacted_at);
CREATE INDEX idx_tx_user_subcat ON transactions (user_id, sub_id);

-- 미션 템플릿 (이미 구축한 테이블과 동일 컨셉)
CREATE TABLE IF NOT EXISTS mission_templates (
  template_id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(64) NOT NULL,                        -- 예: BAN_CATEGORY_DAILY
  render_str TEXT NOT NULL,                         -- "오늘 {label} 안 사기"
  placeholders JSON NOT NULL,                       -- {label:{type:'category_slug' ...}, ...}
  dsl_skeletons JSON NOT NULL,                      -- 스켈레톤들
  template_version INT NOT NULL DEFAULT 1,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 사용자별 미션 인스턴스
CREATE TABLE IF NOT EXISTS mission_instances (
  mission_id INT PRIMARY KEY AUTO_INCREMENT,
  user_id INT NOT NULL,
  template_id INT NOT NULL,
  category_slug VARCHAR(100) NOT NULL,
  params JSON NOT NULL,
  mission_dsl JSON NOT NULL,
  compiled_plan JSON NOT NULL,
  mission TEXT NOT NULL,
  mission_status ENUM('성공','대기','실패') DEFAULT '성공',
  time_tag INT NOT NULL, -- 1: 데일리, 2: 주간, 3: 월간
  valid_from DATE NOT NULL,
  valid_to DATE  NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_mi_user_status (user_id, status),
  INDEX idx_mi_user_valid (user_id, valid_from, valid_to)
);

-- 평가 결과(실패 즉시/성공 EOD)
CREATE TABLE IF NOT EXISTS mission_evaluations (
  eval_id INT PRIMARY KEY AUTO_INCREMENT,
  mission_id INT NOT NULL,
  txn_id VARCHAR(64) NOT NULL,
  event_ts TIMESTAMP NOT NULL,
  decision ENUM('success','fail','irrelevant') NOT NULL,
  explain_json JSON,
  savings_estimated INT DEFAULT 0,
  UNIQUE KEY uq_mission_txn (mission_id, txn_id),
  INDEX idx_ev_mission_time (mission_id, event_ts)
);

-- 일일 누적 카운터(러닝 토탈)
CREATE TABLE IF NOT EXISTS daily_user_category_stats (
  date_kst DATE NOT NULL,
  user_id INT NOT NULL,
  category_slug VARCHAR(100) NOT NULL,
  txn_count INT NOT NULL DEFAULT 0,
  amount_sum INT NOT NULL DEFAULT 0,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (date_kst, user_id, category_slug)
);
