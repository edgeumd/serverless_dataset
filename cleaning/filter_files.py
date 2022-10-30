

class SlsCleaner:
    def __init__(self,sls_list) -> None:
        self.sls_list = set(sls_list)

        self.processed_sls_list = self.process()


    def filter_files(self,sls_path):
        """
        Removes the filter from the given complete sls path
        e.g. www.github.com/user/repo/serverless_folder
    
        params:
        sls_path: Complete sls path

        returns:
        True if it is good
        False if it is bad
        """
        filters = {"learn","sample","hello","greeting","template","example","test","demo","github.com/serverless", "starter","basic","course", "github.com/Azure", "github.com/aws"}
        # github.com/aws is there so github.com/aws-samples will also be removed
        # filters = {"sample","template","example","test","demo","github.com/serverless", "starter"}
        # todo can also be considered as a filter.
        for word in filters:
            sls_path = sls_path.lower()
            if word in sls_path:
                return False
        return True

#bot wasm iot api workshop lab course,

    def make_links(self,sls_raw, root="https://www.github.com/"):
        """
        Makes the link from the given sls_raw
        e.g. from damjee/huu-template to www.github.com/damjee/huu-template
        """
        return root+sls_raw

    
    def process(self):
        """
        Processes the given sls_list
        """
        new_sls_list = []
        new_sls_links= []
        for sls in self.sls_list:
            new_sls_links.append(self.make_links(sls))
            if self.filter_files(new_sls_links[-1]):
                new_sls_list.append(new_sls_links[-1])
        
        return new_sls_list



    