def test_func():
    request_mock = MagicMock()
    request_mock.get.retutn_valeu = "User_id"
    
    with patch("calculate_score", retrurn_value=100), \
            patch("fetch_user_data", request_mock):
                reult = get_user_score("username")
    assert result == 100
         
