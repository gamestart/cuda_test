#  FILENAME:      CMakeLists.txt
#
#  AUTHORS:       Wang Xiaofei    START DATE: Tuesday March 9th 2021
#
#  LAST MODIFIED: Sunday, April 25th 2021, 11:02:41 am
#
#  CONTACT:       xiaofei.wang@smartmore.com

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
INTERFACE
    .
)
