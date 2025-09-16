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
    is_target BOOLEAN NOT NULL DEFAULT TRUE,
    FOREIGN KEY (major_id) REFERENCES major_categories(major_id) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS transactions (
	transaction_id INT AUTO_INCREMENT PRIMARY KEY, -- 결제내역 아이디
	user_id		INT	NULL,             -- 유저 아이디
	sub_id		INT	NOT NULL,         -- 소분류 카테고리 
	-- major_id 	INT NOT NULL,         -- 대분류 카테고리
	transacted_at	TIMESTAMP	NULL, -- 거래 시간
	amount	INT	NULL,             -- 금액
	merchanr_name	VARCHAR(255)	NULL,  -- 거래처
	status	enum('반영', '미반영')	NULL,      -- 반영 / 미반영
	created_at	TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,      -- 생성일
	updated_at	TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,       -- 수정일
    
	FOREIGN KEY (sub_id) REFERENCES sub_categories(sub_id) ON UPDATE CASCADE ON DELETE RESTRICT,
    INDEX idx_tx_user_time (user_id, transacted_at),
	INDEX idx_tx_user_subcat (user_id, sub_id)
);

-- 미션 템플릿 (이미 구축한 테이블과 동일 컨셉)
CREATE TABLE IF NOT EXISTS mission_templates (
  template_id 	INT PRIMARY KEY AUTO_INCREMENT,
  name 			VARCHAR(100) NOT NULL,                -- 예: BAN_CATEGORY_DAILY
  mission 		VARCHAR(200) NOT NULL,                -- "오늘 {label} 안 사기"
  placeholders 	JSON NOT NULL,                        -- { } 설명
  dsl 			JSON NOT NULL,                        --
  version 		INT NOT NULL DEFAULT 1,
  created_at 	TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 사용자별 미션 저장소 
CREATE TABLE IF NOT EXISTS mission (
  mission_id 	INT PRIMARY KEY AUTO_INCREMENT,
  user_id 		INT NOT NULL,
  template_id 	INT NOT NULL,
  sub_id		INT NOT NULL,       -- sub_category name 
  params 		JSON NOT NULL DEFAULT (JSON_OBJECT()),
  dsl 			JSON NOT NULL,
  compiled_plan JSON NOT NULL,  -- 미션 달성 조건 
  mission 		TEXT NOT NULL,
  mission_status ENUM('성공','대기','실패') DEFAULT '대기',
  -- time_tag 		INT NOT NULL,     -- 1: 데일리, 2: 주간, 3: 월간
  time_tag       ENUM('데일리','주간','월간') NOT NULL,
  valid_from 	DATE NOT NULL, 		-- 미션 유호 시작일
  valid_to 		DATE  NOT NULL,  	-- 미션 유효 마감일
  created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  
  -- TODO : user_id FK로 바꾸기 
  -- FOREIGN KEY (user_id) REFERENCES user(user_id),
  FOREIGN KEY (template_id) REFERENCES mission_templates(template_id) ON UPDATE CASCADE ON DELETE RESTRICT,
  FOREIGN KEY (sub_id) REFERENCES sub_categories(sub_id) ON UPDATE CASCADE ON DELETE RESTRICT,

  INDEX idx_mi_user_status (user_id, mission_status),
  INDEX idx_mi_user_valid (user_id, valid_from, valid_to)
);

