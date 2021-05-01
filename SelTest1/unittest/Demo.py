from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
driver = webdriver.Chrome('/Users/honghangyeol/Downloads/chromedriver-2')
driver.implicitly_wait(10)
driver.get("http://google.com")
driver.find_element_by_name("q").send_keys("Hongik University")
driver.find_element_by_name("btnK").click()
driver.maximize_window()
driver.refresh()
time.sleep(2)
driver.quit()
print("")
print("")
print("Test Completed Successfully.")