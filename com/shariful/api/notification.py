# from txfcm import TXFCMNotification
# from twisted.internet import reactor
#
# push_service = TXFCMNotification(api_key="<api-key>")
#
# # Your api-key can be gotten from:  https://console.firebase.google.com/project/<project-name>/settings/cloudmessaging
# # Send to multiple devices by passing a list of ids.
# registration_ids = ["<device.csv registration_id 1>", "<device.csv registration_id 2>", ...]
# message_title = "Uber update"
# message_body = "Hope you're having fun this weekend, don't forget to check today's news"
# df = push_service.notify_multiple_devices(registration_ids=registration_ids, message_title=message_title,
#                                           message_body=message_body)
#
#
# def got_result(result):
#     print (result)
#
#
# df.addBoth(got_result)
# reactor.run()