var twitter = require('twitter');

var tapi = new twitter({
  consumer_key: process.env.TWITTER_CONSUMER_KEY,
  consumer_secret: process.env.TWITTER_CONSUMER_SECRET,
  bearer_token: process.env.TWITTER_BEARER_TOKEN
});

tapi.get('statuses/user_timeline', {screen_name: process.argv.slice(2)[0], count: '200', include_rts: 'false'}, function(error, tweets, response){
      var tweets_text = {"tweets": []}
      for(var i = 0; i < tweets.length; i++)
      {
      console.log(tweets[i].text);
      }

})