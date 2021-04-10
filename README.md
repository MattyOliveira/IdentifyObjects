# IdentifyObjects

IdentifyObjects is a library to assist classifier models in android.

### Version
##### Gradle
```
implementation 'com.github.mattyoliveira:identify-objects:1.0.0'
```

##### Maven
```
<dependency>
  <groupId>com.github.mattyoliveira</groupId>
  <artifactId>identify-objects</artifactId>
  <version>1.0.0</version>
  <type>aar</type>
</dependency>
```

### Example

##### Setup Classifier
```kotlin
class ExempleActivity : AppCompatActivity() {

    private var classifier: Classifier? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.exemple_activity)
        
        Classifier(
              //Set assets path
                assets, 
              //Set model path
                modelPath, 
              //Set label path
                labelPath
            )
    }
	
}
```

<br/>

### Technology

IdentifyObjects use the TensorFlowLite library.

### License
[Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
