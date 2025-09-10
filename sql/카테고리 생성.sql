use pocketc;

-- 고정비
INSERT INTO major_categories (major_name, major_type) VALUES
('주거/생활', '고정비'),
('통신/인터넷', '고정비'),
('금융/저축', '고정비');

-- 변동비
INSERT INTO major_categories (major_name, major_type) VALUES
('식비', '변동비'),
('교통/차량', '변동비'),
('건강/의료', '변동비'),
('교육/자기계발', '변동비'),
('쇼핑', '변동비'),
('문화/여가', '변동비'),
('기타', '변동비');


-- 세분류 
INSERT INTO sub_categories (major_id, sub_name, is_target) VALUES
(4, '커피', TRUE),
(4, '음료(반복 구매)', TRUE),
(4, '술', TRUE),
(4, '간식', TRUE),
(4, '배달음식', TRUE),
(4, '외식', TRUE),
(4, '식재료', TRUE);

INSERT INTO sub_categories (major_id, sub_name, is_target) VALUES
(5, '대중교통', TRUE),
(5, '택시/대리', TRUE),
(5, '유류비', TRUE);

INSERT INTO sub_categories (major_id, sub_name, is_target) VALUES
(1, '월세/관리비', FALSE),
(1, '전기세', FALSE),
(1, '수도세', FALSE),
(1, '가스비', FALSE),
(1, '생활용품', TRUE);

INSERT INTO sub_categories (major_id, sub_name, is_target) VALUES
(2, '휴대폰 요금', FALSE),
(2, '인터넷', FALSE),
(2, 'OTT/구독서비스', TRUE);

INSERT INTO sub_categories (major_id, sub_name, is_target) VALUES
(6, '병원', FALSE),
(6, '약국', FALSE),
(6, '헬스/PT', TRUE);

INSERT INTO sub_categories (major_id, sub_name, is_target) VALUES
(7, '학원/수강료', FALSE),
(7, '도서/교재', TRUE),
(7, '자격증', FALSE);

INSERT INTO sub_categories (major_id, sub_name, is_target) VALUES
(8, '의류/패션', TRUE),
(8, '뷰티/미용', TRUE),
(8, '온라인 쇼핑몰', TRUE),
(8, '충동구매', TRUE);

INSERT INTO sub_categories (major_id, sub_name, is_target) VALUES
(9, '영화/공연', TRUE),
(9, '게임/콘텐츠', TRUE),
(9, '여행', TRUE),
(9, '취미/오락', TRUE);

INSERT INTO sub_categories (major_id, sub_name, is_target) VALUES
(3, '저축', FALSE),
(3, '투자', FALSE),
(3, '대출/이자', FALSE),
(3, '세금/보험', FALSE);

INSERT INTO sub_categories (major_id, sub_name, is_target) VALUES
(10, '송금', FALSE),
(10, '경조사비', FALSE),
(10, '기타', TRUE);
