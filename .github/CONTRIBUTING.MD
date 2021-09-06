## ***Contribution Guide***


Contributions are always welcome, it would be great to have people use and contribute to this project.
***
### How to contribute
* **Create an ISSUE**
    * The first step is to create an issue. It could be a feature request, bug fix, documentation changes,etc
    * Once discussion is done on the issue and approved by the assignees. Please move forward to create a PR.

* **Create a PR**
    * Once the issue is approved create a PR mentioning the issue.
    * If you have implemented something new ensure it is backward compatible.
    * Write all the neccessary tests to maintain code-coverage.

***
#### Steps to Follow
* Git clone the repo.
* ```cd``` into the repo
* Do editable install using ```pip install -e .```
* Install additional dependencies mentioned in ```requirements/test_requirements.txt```
* Also, you need to install pre-commit. What this will do is standardize the code according the code-style
* After you have implemented the feature or bug write unit-tests. Remember use pytest for writing tests.
* If you have implemented a new feature add ```.rst``` file in ```docs/API``` or if its a
    subset of other feature add docs to that ```.rst``` file.
* Once you create a PR you will a ```readthedocs``` test being run and you will be able to see the rendered docs.
***
**NOTE** : We are more than happy to help you on your PRs. So do not hesistate to contact us.
