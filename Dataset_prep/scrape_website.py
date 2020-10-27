# Author: Raymond Wang <raymondwang@u.northwestern.edu>
# Functionality: scrape alloy steel data from MatWeb and save to csv files

from selenium import webdriver
from selenium.webdriver.support.ui import Select
import pandas as pd
import numpy as np
import time
import sys
import os.path
import os

# navigate to target website
options = webdriver.ChromeOptions()
options.add_argument("headless") # comment this for visualization
driver_path = os.getcwd() + "/chromedriver-linux" # google-chrome driver for Linux systems
#driver_path = os.getcwd() + "/chromedriver-mac" # google-chrome driver for macOS
driver = webdriver.Chrome(executable_path=driver_path, chrome_options=options)
driver.get("http://www.matweb.com/index.aspx")

# user login, less likely to be blocked
driver.find_element_by_link_text("LOG IN").click()
time.sleep(2)
username = driver.find_element_by_id("ctl00_ContentMain_txtUserID")
username.send_keys("USERNAME")
passwd =  driver.find_element_by_id("ctl00_ContentMain_txtPassword")
passwd.send_keys("PASSWD")
time.sleep(0.5)
driver.find_element_by_xpath("//input[@src='/images/buttons/btnLogin.gif']").click()
time.sleep(3)

# Alloy composition
driver.find_element_by_link_text("Alloy Composition").click()

# unfold "Ferrous Metal"
driver.find_element_by_id("ctl00_ContentMain_ucMatGroupTree_LODCS1_msTreeViewn1").click()
time.sleep(2)

# choose "Alloy Steel"
driver.find_element_by_id("ctl00_ContentMain_ucMatGroupTree_LODCS1_msTreeViewt6").click()
time.sleep(2)

# choose composition1: Mn
select_fe = Select(driver.find_element_by_id("ctl00_ContentMain_ucPropertyDropdown1_drpPropertyList"))
select_fe.select_by_visible_text("Manganese, Mn")
driver.find_element_by_id("ctl00_ContentMain_ucPropertyEdit1_txtpMin").send_keys("0.0")
time.sleep(2)

# choose composition2: Cr
select_cr = Select(driver.find_element_by_id("ctl00_ContentMain_ucPropertyDropdown2_drpPropertyList"))
select_cr.select_by_visible_text("Chromium, Cr")
driver.find_element_by_id("ctl00_ContentMain_ucPropertyEdit2_txtpMin").send_keys("0.0")
time.sleep(2)

# choose composition3: Ni
select_ni = Select(driver.find_element_by_id("ctl00_ContentMain_ucPropertyDropdown3_drpPropertyList"))
select_ni.select_by_visible_text("Nickel, Ni")
driver.find_element_by_id("ctl00_ContentMain_ucPropertyEdit3_txtpMin").send_keys("0.0")
time.sleep(2)

# lauch searching process
driver.find_element_by_id("ctl00_ContentMain_btnSubmit").click()
time.sleep(3)

# show 200 per page
instance_per_page = 200
Select(driver.find_element_by_id("ctl00_ContentMain_UcSearchResults1_drpPageSize1")).select_by_visible_text(str(instance_per_page))
time.sleep(3)

# loop over 5 pages of results to collect data
npages = 5
for ipage in range(npages):
    print("currently on page " + str(ipage+1))
    sys.stdout.flush()

    # save name of datasets to a list before processing, may not be ideal design but avoids stale element problems
    name_list = []
    for item in driver.find_elements_by_xpath("//td[@style='width:auto; font-weight:bold;']"):
        name_list.append(item.text)
    
    for name in name_list:
        # preprocessing
        fname = name.replace(" ", "_")
        fname = fname.replace("/", "_")
        fname = fname.replace(",", "")
        if len(fname) > 240: fname = fname[:240]+"..." # in case length of file name exceeds Linux limit (255B)
        pathname = "data_raw/" + fname + ".csv"
        if os.path.isfile(pathname):
            if os.stat(pathname).st_size == 0: # in case last written unsuccessful (e.g. blocked)
                print("  removing: " + pathname + ".csv")
                os.remove(pathname)
            else:
                print("  skipping: " + pathname + ".csv")
                continue
        print('  processing: \"' + fname + '\"')
        sys.stdout.flush()
        
        # open file
        f = open("data_raw/" + fname + ".csv", 'w')
        
        # navigate into each alloy page
        driver.find_element_by_link_text(name).click()
        time.sleep(5)
        table = driver.find_element_by_xpath("//table[@class='tabledataformat']")
        attrib = []
        for row in table.find_elements_by_xpath("//tr[@class='altrow datarowSeparator']"):
            attrib.append([d.text for d in row.find_elements_by_css_selector('td')])
            time.sleep(0.5)
        for row in table.find_elements_by_xpath("//tr[@class=' datarowSeparator']"):
            attrib.append([d.text for d in row.find_elements_by_css_selector('td')])
            time.sleep(0.5)
        attrib = np.array(attrib)
        
        # write to file
        df = pd.DataFrame(data=attrib[:, 1:], columns=['Metric', 'English', 'Comments'], index=attrib[:, 0] )
        df.to_csv(f)
        f.close()
        driver.back()
        time.sleep(5)

    # navigate to the next page
    driver.find_element_by_id("ctl00_ContentMain_UcSearchResults1_lnkNextPage").click()
    time.sleep(5)

# quit chrome driver
driver.quit()

