if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

mv recognizer.py /bin/
mv templates/ /etc/templates/

crontab -l > cron

echo "@reboot python3 /bin/recognizer.py &" >> cron

crontab cron
rm cron
